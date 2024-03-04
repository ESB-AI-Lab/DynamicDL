'''
PurposeToken module
'''

from abc import abstractmethod

from ..Token import Token

class _PurposeToken(Token):
    '''
    Abstract interface for defining purpose tokens. Each dataset will have a required set of purpose
    tokens that exist within the files.
    '''
    Presets = None

    @abstractmethod
    def __init__(self):
        pass
