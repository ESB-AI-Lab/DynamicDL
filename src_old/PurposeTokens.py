'''
PurposeTokens module
'''

from typing import Self
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
    def __init__(self, mode: str):
        self.mode: str = mode

    def __eq__(self, other: Self) -> bool:
        if self.__class__ != other.__class__:
            return False
        return self.mode == other.mode

    def __repr__(self) -> str:
        return self.mode

class DirectoryPurposeToken(PurposeToken):
    '''
    Represents the purpose of the directory.
    '''
    def __init__(self, mode: str):
        self.mode: str = mode

    def __eq__(self, other: Self) -> bool:
        if self.__class__ != other.__class__:
            return False
        return self.mode == other.mode

    def __repr__(self) -> str:
        return self.mode

class File:
    '''
    File purpose token presets.
    '''
    IMAGE = FilePurposeToken('Image')
    ANNOTATION = FilePurposeToken('Annotation')

class Directory:
    '''
    Directory purpose token presets.
    '''
    IMAGES = DirectoryPurposeToken('Images')
    ANNOTATIONS = DirectoryPurposeToken('Annotations')
    GENERIC = DirectoryPurposeToken('Generic')
