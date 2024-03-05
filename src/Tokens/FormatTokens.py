'''
FormatToken module
'''
from abc import abstractmethod

from .Token import Token
from .StringFormatToken import StringFormatToken
from .StructureTokens import PathToken
from .._utils import union
from ..DataItems import DataTypes, DataItem

class FormatToken(Token):
    '''
    The FormatToken abstract class is a framework for format classes to support utility functions
    for parsing annotation data.
    '''

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_data(self, file: str) -> dict:
        '''
        Retrieve data.
        
        - file (str): path to the annotation file.
        '''

class JSONToken(FormatToken):
    '''
    Utility functions for parsing json files.
    '''

    def __init__(self):
        pass

class TXTToken(FormatToken):
    '''
    Utility functions for parsing txt files.
    '''
    class __FormatParser:
        def __init__(self, line_format: list[StringFormatToken]):
            self.index: int = 0
            self.length: int = len(line_format)
            self.line_format: list[StringFormatToken] = line_format

        def next_format(self) -> StringFormatToken:
            '''Return the next StringFormatToken in queue'''
            self.index += 1
            self.index %= self.length
            return self.line_format[self.index - 1]

        def is_start(self) -> bool:
            '''True if parser is at start of an object iteration'''
            return self.index == 0

        def is_end(self) -> bool:
            '''True if parser is at end of an object iteration'''
            return self.index == self.length - 1

    def __init__(self, line_format: list[StringFormatToken] | StringFormatToken, offset: int = 0,
                 ignore_type: list[StringFormatToken | str] | StringFormatToken | str = None,
                 by_line: bool = True):
        '''
        Initialize the constructor.
        
        - line_format (list[StringFormatToken] | StringFormatToken): the structure to parse, 
                                                                     repetitively, with data.
        - offset (int): number of items to skip from the top of the file.
        - ignore_type (list[StringFormatToken | str] | StringFormatToken | str): ignore the list of
                                                                                 formats or lines
                                                                                 beginning with str
                                                                                 when parsing.
        - by_line (bool): true if parsing is to be done per line, rather than continuously.
        '''
        self.line_format = union(line_format)
        self.offset: int = offset
        self.ignore_type: list[StringFormatToken] = []
        if ignore_type:
            ignore_type = union(ignore_type)
            self.ignore_type = [StringFormatToken(rule + '{}', DataTypes.GENERIC) if
                           isinstance(rule, str) else rule for rule in ignore_type]
        self.by_line: bool = by_line

    def get_data(self, file: PathToken) -> list[list[DataItem]]:
        '''
        Retrieve data, returning in a list of objects which should contain all data about an item.
        
        - file (FileToken): path to the annotation file.
        '''
        with open(file.get_os_path(), 'r', encoding='utf-8') as f:
            lines: list[str] = f.readlines()
        data = []
        parser: self.__FormatParser = self.__FormatParser(self.line_format)
        if self.by_line:
            for line in lines[self.offset:]:
                if any([check.match(line) for check in self.ignore_type]):
                    continue
                if parser.is_start():
                    item: list[DataItem] = []
                pattern: StringFormatToken = parser.next_format()
                item += pattern.match(line, insertion=True)
                if parser.is_end():
                    data.append(item)
        return data

class XLSXToken(FormatToken):
    '''
    Utility functions for parsing xlsx files.
    '''

    def __init__(self):
        pass

class XMLToken(FormatToken):
    '''
    Utility functions for parsing xml files.
    '''

    def __init__(self):
        pass

class YAMLToken(FormatToken):
    '''
    Utility functions for parsing yaml files.
    '''

    def __init__(self):
        pass
