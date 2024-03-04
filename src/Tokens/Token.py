'''
Abstract base class for all Tokens.
'''

from abc import ABC

class Token(ABC):
    '''
    The Token class is the base class for anything rooted in the configuration. Many different types
    of Tokens extend it, and the extensions will have present types.
    '''
    def __init__(self):
        pass
