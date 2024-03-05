'''
PurposeTokens module
'''

from abc import abstractmethod

from .Token import Token

class PurposeToken(Token):
    '''
    Abstract interface for defining purpose tokens. Each dataset will have a required set of purpose
    tokens that exist within the files.
    '''
    @abstractmethod
    def __init__(self):
        pass

class FilePurposeToken(PurposeToken):
    '''
    Represents the purpose of the file.
    '''
    def __init__(self):
        pass

class DirectoryPurposeToken(PurposeToken):
    '''
    Represents the purpose of the directory.
    '''
    def __init__(self):
        pass
