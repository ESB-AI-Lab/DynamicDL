'''
Generic Format Token module for wildcard strings.
'''
import re

from .Token import Token
from .._utils import union
from ..DataItems import DataItem, DataType

class StringFormatToken(Token):
    '''
    A token class which provides utility functions for converting between tokens and placeholders.
    
    Instance variables:
    - tokens (list[DataItem] | DataItem): the list of tokens which correspond to each wildcard {}
    - pattern (str): the pattern for matching
    '''
    def __init__(self, pattern: str, tokens: list[DataType] | DataType):
        self.tokens: list[DataType] = union(tokens)
        assert pattern.count('{}') == len(self.tokens), \
            "Length of tokens must match corresponding wildcards {}"
        self.pattern: str = pattern

    def match(self, entry: str, insertion: bool = False) -> list[DataItem]:
        '''
        Return a list of the tokens' string values provided an entry string which follows the 
        pattern.
        
        - entry (str): the string to match to the pattern, assuming it does match
        - insertion (bool): tolerate new storage tokens
        '''
        pattern: str = self.pattern.replace('{}', '(.*)')
        matches: list[str] = re.findall(pattern, entry)
        result: list[DataItem] = []
        try:
            if not matches:
                return []
            # if multiple token matching, extract first matching; else do nothing
            if isinstance(matches[0], tuple):
                matches = matches[0]
            for data_type, match in zip(self.tokens, matches):
                result.append(DataItem(data_type, match, insertion=insertion))
        except AssertionError:
            print('StringFormatToken Warning: parsed string format did not match data requirements')
            return []
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
