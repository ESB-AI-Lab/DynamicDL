'''
FormatToken module
'''
import re
from abc import ABC, abstractmethod

from .Token import Token, PathToken, FormatToken
from .IdentifierTokens import UniqueToken
from ._utils import union
from .DataItems import DataTypes, DataItem, DataEntry, DataType

class PatternAlias:
    '''
    Class used when a DataType placeholder could be interpreted multiple ways. For example, if
    IMAGE_NAME also contains CLASS_NAME and IMAGE_ID, we can extract all 3 tokens out using
    PatternAlias. Counts for a single wildcard token.
    '''
    def __init__(self, patterns: list[str], aliases: list[list[DataType] | DataType]):
        aliases: list[list[DataType]] = [union(alias) for alias in aliases]
        assert len(patterns) == len(aliases), \
            f'Pattern and alias list lengths ({len(patterns)}, {len(aliases)}) must match'
        for pattern, alias in zip(patterns, aliases):
            count = pattern.count('{}')
            assert count == len(alias), \
                f'Pattern must have as many wildcards ({count}) as aliases ({len(alias)})'
        self.patterns: list[str] = patterns
        self.aliases: list[list[DataType]] = aliases
        self.desc = ''.join([token.desc for alias in self.aliases for token in alias])

    def get_matches(self, entry: str) -> list[DataItem]:
        '''
        Return a list of DataItems including all of the possible alias items.
        '''
        result: list[DataItem] = []
        for pattern, alias in zip(self.patterns, self.aliases):
            pattern: str = pattern.replace('{}', '(.+)')
            matches: list[str] = re.findall(pattern, entry)
            try:
                if not matches:
                    return []
                # if multiple token matching, extract first matching; else do nothing
                if isinstance(matches[0], tuple):
                    matches = matches[0]
                for data_type, match in zip(alias, matches):
                    result.append(DataItem(data_type, match))
            except AssertionError:
                return []
        return result

    def __repr__(self) -> str:
        return str(dict(zip(self.patterns, self.aliases)))

class StringFormatToken(Token):
    '''
    A token class which provides utility functions for converting between tokens and placeholders.
    
    Instance variables:
    - tokens (list[DataItem] | DataItem): the list of tokens which correspond to each wildcard {}
    - pattern (str): the pattern for matching
    '''
    def __init__(self, pattern: str,
                 tokens: list[DataType | PatternAlias] | DataType | PatternAlias):
        self.tokens: list[DataType | PatternAlias] = union(tokens)
        assert pattern.count('{}') == len(self.tokens), \
            "Length of tokens must match corresponding wildcards {}"
        self.pattern: str = pattern

    def match(self, entry: str) -> list[DataItem]:
        '''
        Return a list of the tokens' string values provided an entry string which follows the 
        pattern.
        
        - entry (str): the string to match to the pattern, assuming it does match
        '''
        pattern: str = self.pattern.replace('{}', '(.+)')
        matches: list[str] = re.findall(pattern, entry)
        result: list[DataItem] = []
        if not matches:
            return []
        # if multiple token matching, extract first matching; else do nothing
        if isinstance(matches[0], tuple):
            matches = matches[0]
        for data_type, match in zip(self.tokens, matches):
            if isinstance(data_type, PatternAlias):
                result += data_type.get_matches(match)
            else:
                result.append(DataItem(data_type, match))
        return result

    def substitute(self, values: list[DataItem] | DataItem) -> str:
        '''
        Return the string representation of the values provided string representations for each
        token as a list
        
        - values (list[str] | str): the values of the tokens to replace, in order
        '''
        values: list[DataItem] = union(values)
        assert all([value.delimiter == token for value, token in zip(values, self.tokens)]), \
            f'All supplied values need to match the data type of tokens. Got: \
{[value.delimiter for value in values]}, required: {self.tokens}'
        assert len(values) == len(self.tokens), \
            f'Number of values did not match (expected: {len(self.tokens)}, actual: {len(values)})'
        return self.pattern.format(*[value.value for value in values])

    def __repr__(self) -> str:
        return self.pattern.format(*self.tokens)

    def __hash__(self) -> int:
        return hash(self.pattern + ''.join([token.desc for token in self.tokens]))

    def __eq__(self, other) -> bool:
        if self.__class__ != other.__class__: return False
        return self.pattern == other.pattern and self.tokens == other.tokens

class Generic(ABC):
    '''
    Represents a generic list/dict for JSON/YAML/XML parsing.
    '''
    def __init__(self, form: list | dict):
        self.form: list | dict = form

    def expand(self, items: list[list | dict]) -> list[list | dict]:
        '''
        Select a sublist of items which match the generic format.
        '''
        matches = []
        for item in items:
            if type(item) is not type(self.form): continue
            if not self.matches(item): continue
            matches.append(matches)
        return matches

    @abstractmethod
    def matches(self, item: list | dict) -> bool:
        '''
        Check if keys and deep value types match.
        
        Preconditions: either list or dict, but item and self.form agree
        '''

class JSONToken(FormatToken):
    '''
    Utility functions for parsing json files.
    '''

    def __init__(self, json_format: dict | list):
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

    def get_data(self, file: PathToken) -> tuple[list[DataEntry], list[DataEntry]]:
        '''
        Retrieve data, returning in a list of objects which should contain all data about an item.
        
        - file (FileToken): path to the annotation file.
        '''
        with open(file.get_os_path(), 'r', encoding='utf-8') as f:
            lines: list[str] = f.readlines()
        data: list[DataEntry] = []
        pairing_data: list[DataEntry] = []
        parser: self.__FormatParser = self.__FormatParser(self.line_format)
        if self.by_line:
            for line in lines[self.offset:]:
                if any([check.match(line) for check in self.ignore_type]):
                    continue
                if parser.is_start():
                    item: list[DataItem] = []
                pattern: StringFormatToken = parser.next_format()
                item += pattern.match(line)
                if parser.is_end():
                    entry = DataEntry(item)
                    if entry.unique: data.append(entry)
                    else: pairing_data.append(entry)
        return data, pairing_data

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

def merge_lists(first: list[DataEntry], second: list[DataEntry]) -> list[DataEntry]:
    '''
    Merge two DataEntry lists.
    '''
    data: list[DataEntry] = first.copy()
    unique_identifiers: list[DataType] = [var for var in vars(DataTypes).values() if
                                            isinstance(var, DataType) and
                                            isinstance(var.token_type, UniqueToken)]
    hashmaps: dict[str, dict[str, DataEntry]] = {}
    for identifier in unique_identifiers:
        hashmaps[identifier.desc] = {}
        for entry in data:
            value = entry.data.get(identifier.desc)
            if value: hashmaps[identifier.desc][value.value] = entry

    additional_data: list[DataEntry] = []
    for entry in second:
        merged: bool = False
        for identifier in unique_identifiers:
            value = entry.data.get(identifier.desc)
            if value and value.value in hashmaps[identifier.desc]:
                hashmaps[identifier.desc][value.value].merge(entry)
                merged = True
                break
        if not merged: additional_data.append(entry)
    data += additional_data
    return data

def _convert_str_token(token: DataType | StringFormatToken) -> StringFormatToken:
    if isinstance(token, StringFormatToken): return token
    if isinstance(token, DataType): return StringFormatToken('{}', token)
    raise ValueError('Invalid token provided')
