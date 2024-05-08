'''
Module `processing`

Module with all processing-related data items and algorithms for expand and merge.

File processing classes:
 - `DataFile` (abstract base class)
   - `JSONFile` (json parser)
   - `TXTFile` (txt parser)
   - `XMLFile` (xml parser)
   - `YAMLFile` (yaml parser, untested)

Other classes:
 - `GenericList`
 - `SegmentationObject`
 - `AmbiguousList`
 - `Pairing`

Methods (pseudo-private):
 - `_get_files` (recursive filestructure getter)
 - `_add_to_hashmap` (helper for `_merge_lists`)
 - `_merge_lists` (list merge handler algorithm)
 - `_merge` (high level merge handler after expanding generics)
 - `expand_generics` (expand generics inside annotation files)
 - `expand_file_generics` (expand generics in directory filestructure)
 - `populate_data` (main algorithm)
'''

import os
import json
import heapq
from typing import Any, Union, Optional
from abc import ABC, abstractmethod
import xmltodict
import yaml
from tqdm import tqdm

from ._utils import union, check_map, Warnings
from .data_items import DataTypes, DataItem, Generic, Static, DataType, DataEntry, RedundantToken, \
                        Folder, File, Image, SegmentationImage, UniqueToken

__all__ = [
    'GenericList',
    'SegmentationObject',
    'AmbiguousList',
    'JSONFile',
    'TXTFile',
    'XMLFile',
    'YAMLFile',
    'Pairing'
]

class GenericList:
    '''
    Generic list item. Items inside the list are expected to repeat mod `len(form)`.
    
     - `form` (`list[Any] | Any`): the form to stick to. Each entry in `form` must be some valid
        form following the syntax of `CVData` forms.
    '''
    def __init__(
        self,
        form: Union[list[Any], Any]
    ) -> None:
        self.form = union(form)

    def expand(
        self,
        dataset: list[Any]
    ) -> tuple[dict[Static, Any], list]:
        '''
        Expand list into dict of statics.
        
         - `dataset` (`list[Any]`): the dataset data, which should follow the syntax of `CVData`
            data.
        '''
        if len(dataset) % len(self.form) != 0:
            Warnings.error('generic_list_length', length1=len(dataset), length2=len(self.form))
        item_list: list[Any] = []
        item: list[Static | dict] = []
        pairings = []
        for index, entry in enumerate(dataset):
            result, pairing = expand_generics(entry, self.form[index % len(self.form)])
            pairings += pairing
            item.append(result)
            if (index + 1) % len(self.form) == 0:
                item_list.append(dict(enumerate(item)))
                item = []
        return dict(enumerate(item_list)), pairings

class SegmentationObject:
    '''
    Object to represent a collection of polygonal coordinates for segmentation. Functionally serves
    the purpose of being a wrapper class for `GenericList` and should be instantiated when the only
    contents inside are `DataTypes.X` and `DataTypes.Y` items as well as non-data items.
    
     - `form` (`GenericList | list`): either a GenericList object or a list which will create a GL.
    '''
    def __init__(
        self,
        form: Union[GenericList, list]
    ) -> None:
        if isinstance(form, list):
            form = GenericList(form)
        self.form = form

    @staticmethod
    def _merge(
        data: Union[dict[Union[Static, int], Any], Static]
    ) -> DataEntry:
        '''
        Recursive process for merging data in segmentation object. Differs from main merge algorithm
        because it can only process within unique values.
        
         - `data` (`dict[Static | int, Any] | Static`): the data to merge, from the expand call.
        '''
        # base cases
        if isinstance(data, Static):
            return DataEntry(data.data)
        recursive = []

        # get result
        for key, val in data.items():
            result = SegmentationObject._merge(val)
            # unique entry result
            if isinstance(result, DataEntry):
                if isinstance(key, Static):
                    result = DataEntry.merge(DataEntry(key.data), result)
                if result.unique:
                    recursive.append([result])
                else:
                    recursive.append(result)

        # if inside unique loop, either can merge all together or result has multiple entries
        result = DataEntry([])
        for item in recursive:
            result = DataEntry.merge(item, result)
        return result

    def expand(
        self,
        dataset: list[Any]
    ) -> tuple[dict[Static, Any], list]:
        '''
        Evaluate object by expanding and merging, and extracting the corresponding X, Y values
        which define the SegmentationObject.
        
         - `dataset` (`list[Any]`): the dataset data, which should follow the syntax of `CVData`
            data.
        '''
        item_dict, _ = self.form.expand(dataset)
        entry = self._merge(item_dict)
        entry.data.pop('GENERIC', '')
        x = entry.data.get('X').value
        y = entry.data.get('Y').value
        if x is None or y is None or len(entry.data) != 2:
            Warnings.error( 'invalid_seg_object', keys=", ".join(list(entry.data.keys())))
        if len(x) != len(y):
            Warnings.error(
                'row_mismatch',
                name1='X',
                name2='Y',
                len1=len(x),
                len2=len(y)
            )
        return Static('SegObject', DataItem(DataTypes.POLYGON, list(zip(x, y)))), []

class AmbiguousList:
    '''
    Ambiguous List. Used to represent when an item could either be in a list, or a solo item.
    This is primarily used for XML files.
    
     - `form` (`GenericList | list | Any`): effectively wrapper for GenericList. Either provide with
        GenericList instantiation values or provide a GenericList.
    '''
    def __init__(self, form: Union[GenericList, list, Any]):
        self.form = form if isinstance(form, GenericList) else GenericList(form)

    def expand(self, dataset: Any) -> dict[Static, Any]:
        '''
        Expand potential list into dict of statics.
        
         - `dataset` (`list[Any]`): the dataset data, which should follow the syntax of `CVData`
            data.
        '''
        dataset = union(dataset)
        return self.form.expand(dataset)

class DataFile(ABC):
    '''
    Abstract base class for classes that parse annotation files.
    '''
    @abstractmethod
    def parse(self, path: str) -> dict:
        '''
        Parses a file.
        
         - `path` (`str`): the path to the file.
        '''

class JSONFile(DataFile):
    '''
    Utility functions for parsing json files.
    '''
    def __init__(self, form: dict[Union[Static, Generic], Any]) -> None:
        self.form = form

    def parse(self, path: str) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return expand_generics(data, self.form)

class TXTFile(DataFile):
    '''
    Utility functions for parsing txt files.
    
     - `line_format` (`GenericList | Pairing | Any`): the structure to parse, repetitively. If
        neither a GenericList nor Pairing is passed then structure is assumed to be the args in
        the constructor of a GenericList.
     - `offset` (`int`): number of items to skip from the top of the file.
     - `ignore_type` (`list[Generic | str] | Generic | str`): ignore the list of formats or lines 
        beginning with str when parsing.
     - `by_line` (`bool`): true if parsing is to be done per line, rather than continuously.
    '''
    def __init__(
        self,
        line_format: Union[GenericList, 'Pairing', Any],
        offset: int = 0,
        ignore_type: Union[list[Union[Generic, str]], Generic, str] = None
    ) -> None:
        if not isinstance(line_format, (GenericList, Pairing)):
            line_format = GenericList(line_format)
        self.line_format = line_format
        self.offset: int = offset
        self.ignore_type: list[Generic] = []
        if ignore_type:
            ignore_type = union(ignore_type)
            self.ignore_type = [Generic(rule + '{}', DataTypes.GENERIC) if
                           isinstance(rule, str) else rule for rule in ignore_type]

    def parse(self, path: str) -> list[Static]:
        def filter_ignores(line: str):
            for ignore_type in self.ignore_type:
                if ignore_type.match(line)[0]:
                    return True
            return False
        with open(path, 'r', encoding='utf-8') as f:
            lines: list[str] = f.readlines()[self.offset:]
        filtered_lines = []
        for line in lines:
            if filter_ignores(line):
                continue
            filtered_lines.append(line)
        if isinstance(self.line_format, GenericList):
            return expand_generics(filtered_lines, self.line_format)
        if isinstance(self.line_format, Pairing):
            self.line_format.find_pairings(filtered_lines, in_file=True)
            return {}, [self.line_format]

class XMLFile(DataFile):
    '''
    Utility functions for parsing xml files.
    '''
    def __init__(self, form: dict[Union[Static, Generic], Any]) -> None:
        self.form = form

    def parse(self, path: str) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            data = xmltodict.parse(f.read())
        return expand_generics(data, self.form)

class YAMLFile(DataFile):
    '''
    Utility functions for parsing yaml files.
    '''
    def __init__(self, form: dict[Union[Static, Generic], Any]) -> None:
        self.form = form

    def parse(self, path: str) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return expand_generics(data, self.form)

class Pairing:
    '''
    Used to specify when two nonunique datatypes should be associated together. Most commonly used
    to pair ID and name together.
    - `form` (`Any`): Whatever follows the CVData specified form as required. Pairing is a wrapper
    class so let it behave as it should
    - `paired` (`DataType`): Items which should be associated together.
    
    '''
    def __init__(self, form: Any, *paired: DataType) -> None:
        if len(paired) <= 1:
            Warnings.error('pairings_missing')
        self.paired = set(paired)
        self.paired_desc = {pair.desc for pair in paired}
        self.form = form
        self.redundant = isinstance(paired[0].token_type, RedundantToken)
        if (not all(isinstance(pair.token_type, RedundantToken) for pair in self.paired)
            if self.redundant else
            all(not isinstance(pair.token_type, RedundantToken) for pair in self.paired)):
            Warnings.error('invalid_pairing', paired=', '.join(map(str, paired)))

    def update_pairing(self, entry: DataEntry) -> None:
        '''
        Update a data entry with pairing values, and does nothing if the pairing does not aplpy.
        
         - `entry` (`DataEntry`): the entry to apply this pairing
        '''
        entry_vals = set(entry.data.keys())
        overlap = entry_vals.intersection(self.paired_desc)
        if not overlap:
            return
        to_fill = self.paired_desc - overlap
        overlap = list(overlap)
        if not self.redundant:
            index = self.paired_to_idx.get(overlap[0], {}).get(entry.data[overlap[0]].value)
            if not index:
                return
            for check in overlap:
                res = self.paired_to_idx.get(check, {}).get(entry.data[check].value)
                if not res or res != index:
                    return
            for empty in to_fill:
                entry.data[empty] = self.idx_to_paired[index].data[empty]
            return
        indices = [self.paired_to_idx.get(overlap[0], {}).get(v)
                   for v in entry.data[overlap[0]].value]
        for check in overlap:
            results = [self.paired_to_idx.get(check, {}).get(v) for v in entry.data[check].value]
            if indices != results:
                return
        for empty in to_fill:
            entry.data[empty] = DataItem(getattr(DataTypes, empty),
                                         [self.idx_to_paired[index].data[empty].value[0]
                                          if index is not None else None for index in indices])

    def _find_pairings(self, pairings: dict[Union[Static, int], Any]) -> list[DataEntry]:
        if all(isinstance(key, (Static, int)) and isinstance(val, Static)
                for key, val in pairings.items()):
            data_items = []
            for key, val in pairings.items():
                data_items += key.data + val.data if isinstance(key, Static) else val.data
            return [DataEntry(data_items)]
        pairs = []
        for val in pairings.values():
            if not isinstance(val, Static):
                pairs += self._find_pairings(val)
        return pairs

    def find_pairings(
        self,
        dataset: Any,
        in_file: bool = True,
        path: Optional[str] = None
    ) -> None:
        '''
        Similar to other processes' `expand` function. Finds the pairing values and stores
        the data internally.
        
         - `dataset` (`Any`): the dataset data, which should follow the syntax of `CVData` data.
         - `in_file` (`bool`): distinguisher to check usage of either `expand_generics`
            or `expand_file_generics`.
        '''
        if in_file:
            expanded, _ = expand_generics(dataset, self.form)
        else:
            expanded, _ = expand_file_generics(path, dataset, self.form)
        pairs_try = self._find_pairings(expanded)
        pairs: list[DataEntry] = []
        for pair in pairs_try:
            if self.paired.issubset({item.delimiter for item in pair.data.values()}):
                pairs.append(DataEntry([pair.data[k.desc] for k in self.paired]))
        self.paired_to_idx = {desc.desc: {(v.data[desc.desc].value[0] if self.redundant
                              else v.data[desc.desc].value): i for i, v in enumerate(pairs)}
                              for desc in self.paired}
        self.idx_to_paired = pairs

def expand_generics(
    dataset: Any,
    root: Any
) -> Union[dict, Static]:
    '''
    Expand all generics and replace with statics, inplace.
     - `dataset` (`Any`): the dataset true values, in nested blocks containing values
     - `root` (`Any`): the format of the dataset, in accordance with valid CVData syntax
    '''
    if isinstance(root, list):
        root = GenericList(root)
    if isinstance(root, (GenericList, SegmentationObject)):
        return root.expand(dataset)
    if isinstance(root, DataType):
        root = Generic('{}', root)
    if isinstance(root, Generic):
        success, tokens = root.match(str(dataset))
        if not success:
            Warnings.error('fail_generic_match', dataset=dataset, root=root)
        return Static(str(dataset), tokens), []
    expanded_root: dict[Static, Any] = {}
    generics: list[Generic] = []
    names: set[Static] = set()
    pairings: list[Pairing] = []
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
        if isinstance(key, str):
            key = Static(key)

        # add Static directly to expanded root
        if key.name in dataset:
            names.add(key.name)
            expanded_root[key] = val
            continue
        Warnings.error('static_missing', value=key)

     # expand Generics
    while len(generics) != 0:
        _, _, generic = heapq.heappop(generics)
        generic: Generic
        for name in dataset:
            # basic checks
            if name in names:
                continue
            if isinstance(generic, Folder) and dataset[name] == "File":
                continue
            if isinstance(generic, File) and dataset[name] != "File":
                continue

            # attempt to match name to generic
            status, items = generic.match(name)
            if not status:
                continue
            names.add(name)
            expanded_root[Static(name, items)] = root[generic]

    to_pop = []

    for key, value in expanded_root.items():
        if isinstance(value, list):
            value = GenericList(value)
        if isinstance(value, (dict, Generic)):
            uniques, pairing = expand_generics(dataset[key.name], value)
            expanded_root[key] = uniques
            pairings += pairing
        elif isinstance(value, (GenericList, AmbiguousList)):
            uniques, pairing = value.expand(dataset[key.name])
            expanded_root[key] = uniques
            pairings += pairing
        elif isinstance(value, DataType):
            expanded_root[key] = Static(dataset[key.name], DataItem(value, dataset[key.name]))
        elif isinstance(value, SegmentationObject):
            expanded_root[key] = value.expand(dataset[key.name])[0]
        elif isinstance(value, Pairing):
            value.find_pairings(dataset[key.name], in_file=True)
            pairings.append(value)
            to_pop.append(key)
        else:
            Warnings.error('inappropriate_type', value=value)
    for item in to_pop:
        expanded_root.pop(item)
    return expanded_root, pairings

def _get_files(path: str) -> dict[str, Union[str, dict]]:
    '''Step one of the processing. Expand the dataset to fit all the files.'''
    files: dict[str, Union[str, dict]] = {}
    for file in os.listdir(path):
        if file.startswith('.'):
            continue
        if os.path.isdir(os.path.join(path, file)):
            files[file] = _get_files(os.path.join(path, file))
        else:
            files[file] = "File"
    return files

def expand_file_generics(
    path: str,
    dataset: dict[str, Any],
    root: dict[Union[str, Static, Generic, DataType], Any]
) -> dict:
    '''
    Variant of the expand_generics function above. Also contains path tracking.'
    
     - `path` (`str`): the absolute filepath up to the root/dataset provided
     - `dataset` (`Any`): the dataset true values, in nested blocks containing values
     - `root` (`Any`): the format of the dataset, in accordance with valid CVData syntax
    '''
    expanded_root: dict[Static, Any] = {}
    generics: list[Generic] = []
    names: set[Static] = set()
    pairings: list[Pairing] = []

    # move Statics to expanded root, move Generics to priority queue for expansion
    for i, key in enumerate(root):
        # convert DataType to Generic with low priority
        if isinstance(key, DataType):
            heapq.heappush(generics, (0, i, key))
            continue

        # priority queue push to prioritize generics with the most wildcards for disambiguation
        if isinstance(key, Generic):
            heapq.heappush(generics, (-len(key.data), i, key))
            continue
        val = root[key]

        # convert str to Static
        if isinstance(key, str):
            key = Static(key)

        # add Static directly to expanded root
        if key.name in dataset:
            names.add(key.name)
            expanded_root[key] = val
            continue
        Warnings.error('static_missing', value=key)

    # expand Generics
    while len(generics) != 0:
        _, _, generic = heapq.heappop(generics)
        generic: Generic
        for name in dataset:
            # basic checks
            if name in names:
                continue
            if isinstance(generic, Folder) and dataset[name] == "File":
                continue
            if isinstance(generic, File) and dataset[name] != "File":
                continue

            # attempt to match name to generic
            if isinstance(generic, DataType):
                status, items = Generic('{}', generic).match(name)
            else: status, items = generic.match(name)

            if not status:
                continue
            names.add(name)
            expanded_root[Static(name, items)] = root[generic]
    to_pop = []
    # all items are statics, now process values
    for key, value in expanded_root.items():
        if isinstance(value, dict):
            next_path: str = os.path.join(path, key.name)
            uniques, pairing = expand_file_generics(
                next_path,
                dataset[key.name],
                value
            )
            expanded_root[key] = uniques
            pairings += pairing
        elif isinstance(value, DataFile):
            uniques, pairing = value.parse(os.path.join(path, key.name))
            expanded_root[key] = uniques
            pairings += pairing
        elif isinstance(value, Image):
            expanded_root[key] = Static(
                'Image',
                DataItem(DataTypes.ABSOLUTE_FILE, os.path.join(path, key.name))
            )
        elif isinstance(value, SegmentationImage):
            expanded_root[key] = Static(
                'Segmentation Image',
                DataItem(DataTypes.ABSOLUTE_FILE_SEG, os.path.join(path, key.name))
            )
        elif isinstance(value, Pairing):
            to_pop.append(key)
            value.find_pairings(dataset[key.name], in_file=False, path=os.path.join(path, key.name))
        else:
            Warnings.error('inappropriate_type', value=value)
    for item in to_pop:
        expanded_root.pop(item)
    return expanded_root, pairings

def _add_to_hashmap(hashmaps: dict[str, dict[str, DataEntry]], entry: DataEntry,
                    unique_identifiers: list[DataType]) -> None:
    for id_try in unique_identifiers:
        value = entry.data.get(id_try.desc)
        if not value:
            continue
        if value.value in hashmaps[id_try.desc]:
            hashmaps[id_try.desc][value.value].merge_inplace(entry)
            for id_update in unique_identifiers:
                value_update = entry.data.get(id_update.desc)
                if id_update == id_try or not value_update:
                    continue
                hashmaps[id_update.desc][value_update.value] = hashmaps[id_try.desc][value.value]
            break
        hashmaps[id_try.desc][value.value] = entry

def _merge_lists(lists: list[list[DataEntry]]) -> list[DataEntry]:
    if len(lists) == 0:
        return []

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
    # base cases
    if isinstance(data, Static):
        return DataEntry(data.data)
    if len(data) == 0:
        return []
    recursive = []

    # get result
    for key, val in data.items():
        result = _merge(val)
        # unique entry result
        if isinstance(result, DataEntry):
            if isinstance(key, Static):
                result = DataEntry.merge(DataEntry(key.data), result)
            if result.unique:
                recursive.append([result])
            else: recursive.append(result)
            continue
        # list entry result
        if isinstance(key, Static):
            for item in result:
                item.apply_tokens(key.data)
        recursive.append(result)
    lists: list[list[DataEntry]] = []
    tokens: list[DataEntry] = []
    for item in recursive:
        if isinstance(item, list):
            lists.append(item)
        else:
            tokens.append(item)

    # if outside unique loop, merge lists and apply tokens as needed
    if lists:
        result = _merge_lists(lists)
        if tokens:
            for item in result:
                for token in tokens:
                    item.apply_tokens(token)
        return result

    # if inside unique loop, either can merge all together or result has multiple entries
    if not check_map((item.unique for item in tokens), 2):
        result = DataEntry([])
        for item in tokens:
            result = DataEntry.merge(item, result)
        return result

    # if there are non unique entries then the data will naturally fall through as pairings are
    # meant to catch the nonunique data
    uniques: list[DataEntry] = []
    others: list[DataEntry] = []
    for item in tokens:
        if item.unique:
            uniques.append(item)
        else:
            others.append(item)
    for other in others:
        for entry in uniques:
            entry.apply_tokens(other.data)
    return uniques

def populate_data(root: str, form: dict) -> list[DataEntry]:
    '''
    Parent process for parsing algorithm.
    
     - `root` (`str`): the file path of the root of the data
     - `form` (`dict`): the form of the data, in accordance with CVData syntax.
    '''
    with tqdm(total=4, desc="Getting files", unit="step") as pbar:
        dataset = _get_files(root)
        pbar.update(1)
        pbar.set_description('Expanding generics')
        data, pairings = expand_file_generics(root, dataset, form)
        pbar.update(1)
        pbar.set_description('Merging data')
        data = _merge(data)
        pbar.update(1)
        pbar.set_description('Applying pairing entries')
        for pairing in pairings:
            for entry in data:
                pairing.update_pairing(entry)
        pbar.update(1)
        pbar.set_description('Done!')
    return _merge_lists([data])
