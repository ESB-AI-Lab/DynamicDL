'''
FormatToken module
'''
from abc import abstractmethod

from .Token import Token

class FormatToken(Token):
    '''
    The FormatToken abstract class is a framework for format classes to support utility functions
    for parsing annotation data.
    '''

    @abstractmethod
    def __init__(self):
        pass

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

    def __init__(self):
        pass

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
