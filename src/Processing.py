'''
File processing module.
'''
import json

from typing import Any, Union

from ._utils import union
from .DataItems import DataTypes, DataItem, Generic, Static, expand_generics

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

    def __init__(self, line_format: Union[list[Generic], Generic], offset: int = 0,
                 ignore_type: Union[list[Union[Generic, str]], Generic, str] = None,
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

    def parse(self, path: str) -> list[Static]:
        '''
        Retrieve data, returning in a list of objects which should contain all data about an item.
        
        - path (str): path to the annotation file.
        '''
        with open(path, 'r', encoding='utf-8') as f:
            lines: list[str] = f.readlines()
        data: list[Static] = []
        parser: self.__FormatParser = self.__FormatParser(self.line_format)
        if self.by_line:
            for index, line in enumerate(lines[self.offset:]):
                if any([check.match(line)[0] for check in self.ignore_type]):
                    continue
                parser.next_format(line)
                # may need to change
                if parser.is_end():
                    data.append(Static(str(index), parser.data))
        return data
