'''
Data population & processing algorithms for CVData.
'''
import os
import heapq
from typing import Any, Union

from .data_items import DataEntry, DataItem, DataTypes, DataType, UniqueToken, Static, Generic, \
                       Image, SegmentationImage, Folder, File
from .processing import DataFile, Pairing
from ._utils import get_str

def _get_files(path: str) -> dict[str, Union[str, dict]]:
    '''Step one of the processing. Expand the dataset to fit all the files.'''
    files: dict[str, Union[str, dict]] = {}
    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            files[file] = _get_files(os.path.join(path, file))
        else:
            files[file] = "File"
    return files

def _expand_generics(path: str, dataset: dict[str, Any],
                     root: dict[Union[str, Static, Generic, DataType], Any]) -> dict:
    '''Expand all generics and set to statics within filestructure.'''
    expanded_root: dict[Static, Any] = {}
    generics: list[Generic] = []
    names: set[Static] = set()
    pairings: list[Pairing] = []

    # move Statics to expanded root, move Generics to priority queue for expansion
    for i, key in enumerate(root):
        # convert DataType to Generic with low priority
        if isinstance(key, DataType):
            heapq.heappush(generics, (0, i, Generic('{}', key)))

        # priority queue push to prioritize generics with the most wildcards for disambiguation
        if isinstance(key, Generic):
            heapq.heappush(generics, (-len(key.data), i, key))
            continue
        val = root[key]

        # convert str to Static
        if isinstance(key, str): key = Static(key)

        # add Static directly to expanded root
        if key.name in dataset:
            names.add(key.name)
            expanded_root[key] = val
            continue
        raise ValueError(f'Static value {key} not found in dataset')

    # expand Generics 
    while len(generics) != 0:
        _, _, generic = heapq.heappop(generics)
        generic: Generic
        for name in dataset:
            # basic checks
            if name in names: continue
            if isinstance(generic, Folder) and dataset[name] == "File": continue
            if isinstance(generic, File) and dataset[name] != "File": continue

            # attempt to match name to generic
            status, items = generic.match(name)
            if not status: continue
            names.add(name)
            expanded_root[Static(name, items)] = root[generic]
    to_pop = []
    # all items are statics, now process values 
    for key, value in expanded_root.items():
        if isinstance(value, dict):
            next_path: str = os.path.join(path, key.name)
            uniques, pairing = _expand_generics(next_path, dataset[key.name], expanded_root[key])
            expanded_root[key] = uniques
            pairings += pairing
        elif isinstance(value, DataFile):
            uniques, pairing = value.parse(os.path.join(path, key.name))
            expanded_root[key] = uniques
            pairings += pairing
        elif isinstance(value, Image):
            expanded_root[key] = Static('Image', DataItem(DataTypes.ABSOLUTE_FILE,
                                                           os.path.join(path, key.name)))
        elif isinstance(value, SegmentationImage):
            expanded_root[key] = Static('Segmentation Image', DataItem(DataTypes.ABSOLUTE_FILE_SEG,
                                                           os.path.join(path, key.name)))
        elif isinstance(value, Pairing):
            to_pop.append(key)
            value.find_pairings(dataset[key.name])
        else: 
            raise ValueError(f'Unknown value found in format: {value}')
    for item in to_pop: expanded_root.pop(item)
    return expanded_root, pairings

def _add_to_hashmap(hashmaps: dict[str, dict[str, DataEntry]], entry: DataEntry,
                    unique_identifiers: list[DataType]) -> None:
    '''
    Helper method for _merge_lists(), adds an item to all corresponding hashmaps and handles merge.
    '''
    for id_try in unique_identifiers:
        value = entry.data.get(id_try.desc)
        if not value: continue
        if value.value in hashmaps[id_try.desc]:
            result = hashmaps[id_try.desc][value.value].merge_inplace(entry)
            if not result: raise ValueError(f'Found conflicting information when merging \
                {hashmaps[id_try.desc][value.value]} and {entry}')
            for id_update in unique_identifiers:
                value_update = entry.data.get(id_update.desc)
                if id_update == id_try or not value_update: continue
                hashmaps[id_update.desc][value_update.value] = hashmaps[id_try.desc][value.value]
            break
        hashmaps[id_try.desc][value.value] = entry

def _merge_lists(lists: list[list[DataEntry]]) -> list[DataEntry]:
    '''
    Merge two DataEntry lists.
    '''
    if len(lists) == 0: return []

    # get all unique identifiers
    unique_identifiers: list[DataType] = [var for var in vars(DataTypes).values() if
        isinstance(var, DataType) and isinstance(var.token_type, UniqueToken)]
    
    # append to hashmaps for efficient merge
    hashmaps: dict[str, dict[str, DataEntry]] = {id.desc:{} for id in unique_identifiers}
    for next_list in lists:
        for entry in next_list:
            _add_to_hashmap(hashmaps, entry, unique_identifiers)

    # extract data from all hashmaps, same entries have same pointer so set works for unique items
    data = set()
    for identifier in unique_identifiers:
        data.update(hashmaps[identifier.desc].values())
    return list(data)

def _merge(data: Union[dict[Union[Static, int], Any], Static]) -> \
        Union[DataEntry, list[DataEntry]]:
    '''
    Recursive process for merging unique data. 
    Returns DataEntry if within unique item, list otherwise.
    '''
    # base cases
    if isinstance(data, Static): return DataEntry(data.data)
    if len(data) == 0: return []
    recursive = []

    # get result
    for key, val in data.items():
        result = _merge(val)
        # unique entry result
        if isinstance(result, DataEntry):
            if isinstance(key, Static): result = DataEntry.merge(DataEntry(key.data), result)
            if isinstance(result, DataEntry) and result.unique: recursive.append([result])
            else: recursive.append(result)
            continue
        # list entry result
        if isinstance(key, Static):
            for item in result: item.apply_tokens(key.data)
        recursive.append(result)
    lists = [item for item in recursive if isinstance(item, list)]
    tokens = [item for item in recursive if not isinstance(item, list)]

    # if outside unique loop, merge lists and apply tokens as needed
    if lists:
        result = _merge_lists(lists)
        if tokens: (item.apply_tokens(token) for token in tokens for item in result)
        return result

    # if inside unique loop, either can merge all together or result has multiple entries
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

def populate_data(root, form) -> list[DataEntry]:
    dataset = _get_files(root)
    data, pairings = _expand_generics(root, dataset, form)
    data = _merge(data)
    for pairing in pairings:
        for entry in data:
            pairing.update_pairing(entry)
    return _merge_lists([data])