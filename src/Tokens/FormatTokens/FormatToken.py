'''
FormatToken module
'''
from abc import abstractmethod

from ..Token import Token

class FormatToken(Token):
    '''
    The FormatToken abstract class is a framework for format classes to support utility functions
    for parsing annotation data.
    '''

    @abstractmethod
    def __init__(self):
        pass
