import os
import heapq
from typing import Any
import json
from pandas import DataFrame

from .DataItems import DataEntry, DataItem, DataTypes, merge_lists
from .Names import Static, Generic
from .Processing import TXTFile, JSONFile, Image

def _get_files(path: str) -> dict:
    files = {}
    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            files[file] = _get_files(os.path.join(path, file))
        else:
            files[file] = "File"
    return files

def _expand_generics(path: str, dataset: dict[str, Any],
                     root: dict[str | Static | Generic, Any]) -> dict:
    '''
    Expand all generics and set to statics.
    '''
    expanded_root: dict[Static, Any] = {}
    generics: list[Generic] = []
    names: set[Static] = set()
    for key in root:
        if isinstance(key, Generic):
            # priority queue push to prioritize generics with the most wildcards for disambiguation
            heapq.heappush(generics, (-len(key.data), key))
            continue
        if isinstance(key, str):
            if key in dataset:
                names.add(key)
                expanded_root[Static(key)] = root[key]
            else: raise ValueError(f'Static value {key} not found in dataset')
            continue
        if key.name in dataset:
            names.add(key.name)
            expanded_root[key] = root[key]
        else:
            raise ValueError(f'Static value {key} not found in dataset')

    while len(generics) != 0:
        _, generic = heapq.heappop(generics)
        generic: Generic
        for name in dataset:
            if name in names: continue
            status, items = generic.match(name)
            if not status: continue
            new_name: str = generic.substitute(items)
            names.add(new_name)
            expanded_root[Static(new_name, items)] = root[generic]

    for key, value in expanded_root.items():
        if isinstance(value, dict):
            expanded_root[key] = _expand_generics(os.path.join(path, key.name),
                                                  dataset[key.name], expanded_root[key])
        elif isinstance(value, (TXTFile, JSONFile)):
            expanded_root[key] = value.parse(os.path.join(path, key.name))
        elif isinstance(value, Image):
            expanded_root[key] = Static('Image', DataItem(DataTypes.ABSOLUTE_FILE, 
                                                           os.path.join(path, key.name)))
    return expanded_root

def _make_uniform(data: dict[Static, Any] | list[Any] | Static) -> dict:
    if isinstance(data, dict):
        items = {}
        for key, val in data.items():
            items[key] = _make_uniform(val)
        return items
    elif isinstance(data, list): return {i:_make_uniform(item) for i, item in enumerate(data)}
    return data

def _split(data: dict[Static | int, Any] | Static) -> tuple[dict, dict]:
    if isinstance(data, Static):
        return (None, data) if DataEntry(data.data).unique else (data, None)
    count = 0
    pairings = {}
    uniques = {}
    for key, val in data.items():
        if isinstance(key, Static) and DataEntry(key.data).unique:
            count += 1
            uniques[key] = val
            continue
        if isinstance(val, Static) and DataEntry(val.data).unique:
            count += 1
            uniques[key] = val
            continue
        pairs, unique_vals = _split(val)
        if pairs: pairings[key] = pairs
        if unique_vals: uniques[key] = unique_vals
    if count == 1:
        return (None, data)
    return pairings, uniques

def _find_pairings(pairings: dict[Static | int, Any], curr_data: list[DataItem]) -> list[DataEntry]:
    if all([isinstance(key, (Static, int)) and isinstance(val, Static) for key, val in pairings.items()]):
        data_items = []
        for key, val in pairings.items():
            data_items += key.data + val.data if isinstance(key, Static) else val.data
        return [DataEntry(data_items + curr_data)]
    pairs = []
    structs = []
    for key, val in pairings.items():
        if isinstance(key, Static): curr_data += key.data

        if isinstance(val, Static): curr_data += val.data
        else: structs.append(val)

    for struct in structs:
        pairs += _find_pairings(struct, curr_data)
    return pairs

def _merge(data: dict[Static | int, Any] | Static, pairings: list[DataEntry]) -> \
        DataEntry | list[DataEntry]:
    if isinstance(data, Static):
        entry = DataEntry(data.data) # apply pairings here if possible
        return entry
    if len(data) == 0: return []
    recursive = []
    for key, val in data.items():
        result = _merge(val, pairings)
        if isinstance(result, DataEntry):
            recursive.append(DataEntry.merge(DataEntry(key.data), result)
                             if isinstance(key, Static) else result)
            continue
        if isinstance(key, Static):
            for item in result:
                if isinstance(item, list):
                    for obj in item: obj.apply_tokens(key.data)
                else: item.apply_tokens(key.data)
        recursive.append(result)
    lists = [item for item in recursive if isinstance(item, list)]
    tokens = [item for item in recursive if not isinstance(item, list)]
    if lists:
        result = merge_lists(lists)
        if tokens:
            for token in tokens:
                for item in result: item.apply_tokens(token)
        return result
    entries = []
    entry = recursive[0]
    for index, item in enumerate(recursive[1:], 1):
        res = DataEntry.merge(entry, item, overlap=False)
        if res:
            entry = res
            continue
        entries.append(entry)
        entry = recursive[index]
    entries.append(entry)
    return entries if len(entries) > 1 else entries[0]

def _get_str(data):
    if isinstance(data, dict):
        return {str(key): _get_str(val) for key, val in data.items()}
    if isinstance(data, list):
        return [_get_str(val) for val in data]
    return str(data)

def get_str(data):
    '''Return pretty print string.'''
    return json.dumps(_get_str(data), indent=4).replace('"', '')

class Dataset:
    def __init__(self, root: str, form: dict[Static | Generic, Any]):
        self.root = root
        self.form = form
        self._expand()

    def _expand(self) -> list[DataEntry]:
        '''
        Expand dataset into dict format
        '''
        dataset = _get_files(self.root)
        data = _expand_generics(self.root, dataset, self.form)
        data = _make_uniform(data)
        pairings, uniques = _split(data)
        pairings = _find_pairings(pairings, [])
        self.data: list[DataEntry] = _merge(uniques, pairings)

    def get_dataframe(self) -> DataFrame:
        return DataFrame([data.data for data in self.data])