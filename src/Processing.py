'''
File processing module.
'''
import json
import heapq
from typing import Any, Union

from ._utils import union
from .DataItems import DataTypes, DataItem, Generic, Static, DataType, DataEntry, RedundantToken

class GenericList:
    '''
    Generic list.
    '''
    def __init__(self, form: Union[list[Any], Any]):
        self.form = union(form)

    def expand(self, dataset: list[Any]) -> dict[Static, Any]:
        '''
        Expand list into dict of statics.
        '''
        assert len(dataset) % len(self.form) == 0, \
                f'List length ({len(dataset)})must be a multiple of length of provided form ({len(self.form)})'
        item_list: list[Any] = []
        item: list[Static | dict] = []
        for index, entry in enumerate(dataset):
            result, pairings = expand_generics(entry, self.form[index % len(self.form)])
            item.append(result)
            if (index + 1) % len(self.form) == 0:
                item_list.append({i: v for i, v in enumerate(item)})
                item = []
        return {i: item for i, item in enumerate(item_list)}, pairings

class SegmentationObject:
    '''
    Object to represent a collection of polygonal coordinates for segmentation.
    '''
    def __init__(self, form: Union[GenericList, list]):
        if isinstance(form, list): form = GenericList(form)
        self.form = form

    def expand(self, dataset: list[Any]) -> dict[Static, Any]:
        '''
        Expand object into dict of statics.
        '''
        item_dict, _ = self.form.expand(dataset)
        x = []
        y = []
        for item in item_dict.values(): # need to future-proof for ordered dict implementations
            for i in item.values():
                assert isinstance(i, Static), f'Unknown item {item} found in segmentation object'
                for data in i.data:
                    if data.delimiter == DataTypes.X: x.append(data.value)
                    elif data.delimiter == DataTypes.Y: y.append(data.value)
                    else: raise ValueError('Unknown item found in segmentation object')
        assert len(x) == len(y), 'Mismatch X and Y coordinates'
        return Static('SegObject', DataItem(DataTypes.POLYGON, list(zip(x, y)))), []


class JSONFile:
    '''
    Utility functions for parsing json files.
    '''
    def __init__(self, form: dict[Union[Static, Generic], Any]):
        self.form = form

    def parse(self, path: str) -> dict:
        '''
        Parses JSON file.
        '''
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return expand_generics(data, self.form)

class TXTFile:
    '''
    Utility functions for parsing txt files.
    '''

    def __init__(self, line_format: Union[GenericList, 'Pairing'], offset: int = 0,
                 ignore_type: Union[list[Union[Generic, str]], Generic, str] = None):
        '''
        Initialize the constructor.
        
        - line_format (list[Generic] | Generic): the structure to parse, 
                                                                     repetitively, with data.
        - offset (int): number of items to skip from the top of the file.
        - ignore_type (list[Generic | str] | Generic | str): ignore the list of
                                                                                 formats or lines
                                                                                 beginning with str
                                                                                 when parsing.
        - by_line (bool): true if parsing is to be done per line, rather than continuously.
        '''
        self.line_format = line_format
        self.offset: int = offset
        self.ignore_type: list[Generic] = []
        if ignore_type:
            ignore_type = union(ignore_type)
            self.ignore_type = [Generic(rule + '{}', DataTypes.GENERIC) if
                           isinstance(rule, str) else rule for rule in ignore_type]

    def parse(self, path: str) -> list[Static]:
        '''
        Retrieve data, returning in a list of objects which should contain all data about an item.
        
        - path (str): path to the annotation file.
        '''
        def filter_ignores(line: str):
            for ignore_type in self.ignore_type:
                if ignore_type.match(line)[0]: return True
            return False
        with open(path, 'r', encoding='utf-8') as f:
            lines: list[str] = f.readlines()[self.offset:]
        filtered_lines = []
        for line in lines:
            if filter_ignores(line): continue
            filtered_lines.append(line)
        if isinstance(self.line_format, GenericList):
            return expand_generics(filtered_lines, self.line_format)
        elif isinstance(self.line_format, Pairing):
            self.line_format.find_pairings(filtered_lines)
            return {}, [self.line_format]

class Pairing:
    '''
    Used to specify when two nonunique datatypes should be associated together. Most commonly used
    to pair ID and name together.
    '''
    def __init__(self, form: Any, *paired: DataType):
        assert len(paired) > 1, 'Pairing must have more than one datatype'
        # assert more checks on datatype restrictions
        self.paired = set(paired)
        self.paired_desc = {pair.desc for pair in paired}
        self.form = form
        self.redundant = isinstance(paired[0].token_type, RedundantToken)
    
    def update_pairing(self, entry: DataEntry) -> None:
        entry_vals = set(entry.data.keys())
        overlap = entry_vals.intersection(self.paired_desc)
        if not overlap: return
        to_fill = self.paired_desc - overlap
        overlap = list(overlap)
        if not self.redundant:
            index = self.paired_to_idx.get(overlap[0], {}).get(entry.data[overlap[0]].value)
            if not index: return
            for check in overlap:
                res = self.paired_to_idx.get(check, {}).get(entry.data[check].value)
                if not res or res != index: return
            for empty in to_fill:
                entry.data[empty] = self.idx_to_paired[index].data[empty]
            return
        indices = [self.paired_to_idx.get(overlap[0], {}).get(v)
                   for v in entry.data[overlap[0]].value]
        for check in overlap:
            results = [self.paired_to_idx.get(check, {}).get(v) for v in entry.data[check].value]
            if indices != results: return
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

    def find_pairings(self, dataset) -> list[DataEntry]:
        expanded, _ = expand_generics(dataset, self.form)
        pairs_try = self._find_pairings(expanded)
        pairs: list[DataEntry] = []
        for pair in pairs_try:
            if self.paired.issubset({item.delimiter for item in pair.data.values()}):
                pairs.append(DataEntry([pair.data[k.desc] for k in self.paired]))
        self.paired_to_idx = {desc.desc: {(v.data[desc.desc].value[0] if self.redundant
                              else v.data[desc.desc].value): i for i, v in enumerate(pairs)}
                              for desc in self.paired}
        self.idx_to_paired = pairs # dict maintains order i think

def expand_generics(dataset: Union[dict[str, Any], Any],
                     root: Union[dict[Any], DataType, Generic]) -> Union[dict, Static]:
    '''
    Expand all generics and set to statics.
    '''
    if isinstance(root, (GenericList, SegmentationObject)):
        return root.expand(dataset)
    if isinstance(root, DataType):
        root = Generic('{}', root)
    if isinstance(root, Generic):
        success, tokens = root.match(str(dataset))
        if not success: raise ValueError(f'Failed to match: {dataset} to {root}')
        return Static(str(dataset), tokens), []
    expanded_root: dict[Static, Any] = {}
    generics: list[Generic] = []
    names: set[Static] = set()
    pairings: list[Pairing] = []
    for key in root:
        if isinstance(key, Generic):
            # push to prioritize generics with the most wildcards for disambiguation
            heapq.heappush(generics, (-len(key.data), key))
            continue
        if isinstance(key, str):
            if key in dataset:
                names.add(key)
                expanded_root[Static(key)] = root[key]
            else:
                raise ValueError(f'Static value {key} not found in dataset')
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

    to_pop = []

    for key, value in expanded_root.items():
        if isinstance(value, (dict, Generic)):
            uniques, pairing = expand_generics(dataset[key.name], expanded_root[key])
            expanded_root[key] = uniques
            pairings += pairing
        elif isinstance(value, GenericList):
            uniques, pairing = value.expand(dataset[key.name])
            expanded_root[key] = uniques
            pairings += pairing
        elif isinstance(value, DataType):
            expanded_root[key] = Static(dataset[key.name], DataItem(value, dataset[key.name]))
        elif isinstance(value, SegmentationObject):
            expanded_root[key] = value.expand(dataset[key.name])[0]
        elif isinstance(value, Pairing):
            value.find_pairings(dataset[key.name])
            pairings.append(value)
            to_pop.append(key)
        else:
            raise ValueError(f'Inappropriate value {value}')
    for item in to_pop: expanded_root.pop(item)
    return expanded_root, pairings