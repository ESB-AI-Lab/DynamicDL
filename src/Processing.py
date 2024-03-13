
import json
import heapq
from typing import Any

from ._utils import union
from .Names import Generic, Static
from .DataItems import DataType, DataTypes, DataItem

class Image:
    '''
    Generic image.
    '''
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "Image"

class GenericList:
    '''
    Generic list.
    '''
    def __init__(self, form: list[Any] | Any):
        self.form = union(form)

    def expand(self, dataset: list[Any]) -> dict[Static, Any]:
        '''
        Expand list into dict of statics.
        '''
        assert len(dataset) % len(self.form) == 0, \
                'List length must be a multiple of length of provided form'
        item_list: list[Static] = []
        item: list[Any] = []
        for index, entry in enumerate(dataset):
            # print(entry)
            # print(self.form[index % len(self.form)])
            item.append(_expand_generics(entry, self.form[index % len(self.form)]))
            # print(item)
            if (index + 1) % len(self.form) == 0: 
                item_list.append(Static(index, item))
                item = []
        return {Static(str(i)): value for i, value in enumerate(item_list)}

class JSONFile:
    '''
    Utility functions for parsing json files.
    '''
    def __init__(self, form: dict[Static | Generic, Any]):
        self.form = form

    def parse(self, path: str) -> dict:
        '''
        Parses JSON file.
        '''
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return _expand_generics(data, self.form)

class TXTFile:
    '''
    Utility functions for parsing txt files.
    '''
    class __FormatParser:
        def __init__(self, line_format: list[Generic]):
            self.index: int = 0
            self.length: int = len(line_format)
            self.line_format: list[Generic] = line_format
            self.data: list[DataItem] = []

        def next_format(self, line: str) -> Generic:
            '''Return the next Generic in queue'''
            if self.is_start(): self.data = []
            self.index += 1
            self.index %= self.length
            success, data = self.line_format[self.index - 1].match(line)
            if not success:
                raise ValueError('Line failed to parse')
            self.data += data
            return self.line_format[self.index - 1]

        def is_start(self) -> bool:
            '''True if parser is at start of an object iteration'''
            return self.index == 0

        def is_end(self) -> bool:
            '''True if parser is at end of an object iteration'''
            return self.index == self.length - 1

    def __init__(self, line_format: list[Generic] | Generic, offset: int = 0,
                 ignore_type: list[Generic | str] | Generic | str = None,
                 by_line: bool = True):
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
        self.line_format = union(line_format)
        self.offset: int = offset
        self.ignore_type: list[Generic] = []
        if ignore_type:
            ignore_type = union(ignore_type)
            self.ignore_type = [Generic(rule + '{}', DataTypes.GENERIC) if
                           isinstance(rule, str) else rule for rule in ignore_type]
        self.by_line: bool = by_line

    def parse(self, path: str) -> dict[int, Static]:
        '''
        Retrieve data, returning in a list of objects which should contain all data about an item.
        
        - path (str): path to the annotation file.
        '''
        with open(path, 'r', encoding='utf-8') as f:
            lines: list[str] = f.readlines()
        data: list[Static] = []
        parser: self.__FormatParser = self.__FormatParser(self.line_format)
        if self.by_line:
            for line in lines[self.offset:]:
                if any([check.match(line)[0] for check in self.ignore_type]):
                    continue
                parser.next_format(line)
                # may need to change
                if parser.is_end():
                    data.append(Static(str(len(data)), parser.data))
        return {Static(str(i)): val for i, val in enumerate(data)}

def _expand_generics(dataset: dict[str, Any] | Any,
                     root: dict[str | Static | Generic, Any] | DataType | Generic) -> dict | Static:
    '''
    Expand all generics and set to statics.
    '''
    if isinstance(root, DataType):
        root = Generic('{}', root)
    if isinstance(root, Generic):
        success, tokens = root.match(str(dataset))
        if not success: raise ValueError(f'Failed to match: {dataset} to {root}')
        return Static(root.substitute(tokens), tokens)
    expanded_root: dict[Static, Any] = {}
    generics: list[Generic] = []
    names: set[Static] = set()
    for key in root:
        if isinstance(key, Generic):
            # push to prioritize generics with the most wildcards for disambiguation
            heapq.heappush(generics, (-len(key.data), key))
            continue
        if isinstance(key, str):
            if key.name in dataset:
                names.add(key)
                expanded_root[Static(key)] = root[key]
            else:
                raise ValueError(f'Static value {key} not found in dataset')
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
            expanded_root[key] = _expand_generics(dataset[key.name],
                                                    expanded_root[key])
        elif isinstance(value, GenericList):
            expanded_root[key] = value.expand(dataset[key.name])
        if isinstance(value, DataType):
            expanded_root[key] = Static(dataset[key.name], [value])
        if isinstance(value, Generic):
            success, tokens = value.match(dataset[key.name])
            if not success: raise ValueError(f'Failed to match: {dataset} to {root}')
            expanded_root[key] = Static(dataset[key.name], tokens)
    return expanded_root
