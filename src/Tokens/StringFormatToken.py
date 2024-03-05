import re

from .Token import Token
from .._utils import union

class StringFormatToken(Token):
    '''
    A token class which provides utility functions for converting between tokens and placeholders.
    
    Instance variables:
    - tokens (list[Token] | Token): the list of tokens which correspond to each wildcard {}
    - pattern (str): the pattern for matching
    '''
    def __init__(self, pattern: str, tokens: list[Token] | Token):
        self.tokens: list[Token] = union(tokens)
        assert pattern.count('{}') == len(self.tokens), \
            "Length of tokens must match corresponding wildcards {}"
        self.pattern: str = pattern

    def match(self, entry: str) -> list[str]:
        '''
        Return a list of the tokens' string values provided an entry string which follows the 
        pattern.
        
        - entry (str): the string to match to the pattern, assuming it does match
        '''
        pattern: str = self.pattern.replace('{}', '(.*)')
        matches: list[str] = re.findall(pattern, entry)
        return matches

    def substitute(self, values: list[str] | str) -> str:
        '''
        Return the string representation of the values provided string representations for each
        token as a list
        
        - values (list[str] | str): the values of the tokens to replace, in order
        '''
        values = union(values)
        assert len(values) == len(self.tokens), \
            f'Number of values did not match (expected: {len(self.tokens)}, actual: {len(values)})'
        return self.pattern.format(*values)

    def __repr__(self) -> str:
        return self.pattern