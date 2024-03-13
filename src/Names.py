'''
Labels module.
'''
import re

from ._utils import union
from .DataItems import DataItem, DataType

class Alias:
    '''
    Class used when a DataType placeholder could be interpreted multiple ways. For example, if
    IMAGE_NAME also contains CLASS_NAME and IMAGE_ID, we can extract all 3 tokens out using
    PatternAlias. Counts for a single wildcard token.
    '''
    def __init__(self, generics: list['Generic']):
        assert len(generics) > 0, 'Must have at least 1 generic in list.'
        self.patterns: list[str] = [generic.name for generic in generics]
        self.aliases: list[tuple[DataType, ...]] = [generic.data for generic in generics]
        self.desc = ''.join([token.desc for alias in self.aliases for token in alias])

    def match(self, entry: str) -> tuple[bool, list[DataItem]]:
        '''
        Return a list of DataItems including all of the possible alias items.
        '''
        result: list[DataItem] = []
        for pattern, alias in zip(self.patterns, self.aliases):
            pattern: str = pattern.replace('{}', '(.+)')
            matches: list[str] = re.findall(pattern, entry)
            try:
                if not matches:
                    return False, []
                # if multiple token matching, extract first matching; else do nothing
                if isinstance(matches[0], tuple):
                    matches = matches[0]
                for data_type, match in zip(alias, matches):
                    result.append(DataItem(data_type, match))
            except AssertionError:
                return False, []
        return True, result
    
    def substitute(self, values: list[DataItem]) -> str:
        return Generic(self.patterns[0], *self.aliases[0]).substitute(values[:len(self.aliases[0])])
    
    def length(self) -> int:
        return len(self.patterns)

    def __repr__(self) -> str:
        return str(dict(zip(self.patterns, self.aliases)))

class Static:
    '''
    Represents an object with a static name. Can contain data.
    '''
    def __init__(self, name: str, data: list[DataItem] = []):
        self.name: str = name
        self.data: list[DataItem] = data

    def match(self, entry: str) -> tuple[bool, list[DataItem]]:
        '''
        Checks if the entry string matches this static item.
        '''
        matched: bool = entry == self.name
        data: list[DataItem] = self.data if matched else []
        return matched, data

    def __repr__(self) -> str:
        return f'{self.name} ({", ".join([str(item) for item in self.data])})'

class Generic:
    '''
    Represents an object with a generic name.
    '''
    def __init__(self, name: str, *data: DataType | Alias):
        assert len(data) == name.count('{}'), 'Format must have same number of wildcards'
        self.name: str = name
        self.data: tuple[DataType | Alias, ...] = data

    def match(self, entry: str) -> tuple[bool, list[DataItem]]:
        '''
        Return a list of the tokens' string values provided an entry string which follows the 
        pattern.
        
        - entry (str): the string to match to the pattern, assuming it does match
        '''
        pattern: str = self.name.replace('{}', '(.+)')
        matches: list[str] = re.findall(pattern, entry)
        result: list[DataItem] = []

        if not matches:
            return False, []
        # if multiple token matching, extract first matching; else do nothing
        try:
            if isinstance(matches[0], tuple):
                matches = matches[0]
            for data_type, match in zip(self.data, matches):
                if isinstance(data_type, Alias):
                    success, matched = data_type.match(match)
                    if not success:
                        return False, []
                    result += matched
                else:
                    result.append(DataItem(data_type, match))
        except AssertionError:
            return False, []
        return True, result

    def substitute(self, values: list[DataItem] | DataItem) -> str:
        '''
        Return the string representation of the values provided string representations for each
        token as a list
        
        - values (list[str] | str): the values of the tokens to replace, in order
        '''
        values: list[DataItem] = union(values)
        substitutions: list[str] = []
        index: int = 0
        for token in self.data:
            if isinstance(token, Alias):
                substitutions.append(token.substitute(values[index:]))
                index += token.length()
            else:
                substitutions.append(values[index].value)
                index += 1
        return self.name.format(*substitutions)

    def __repr__(self) -> str:
        return f'{self.name} | {self.data}'
