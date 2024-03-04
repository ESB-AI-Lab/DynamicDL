'''
GenericStructureTokens module
'''

from abc import abstractmethod

from ..._utils import union
from ..PurposeTokens import DirectoryPurposeToken, FilePurposeToken
from .StructureTokens import StructureToken, FileToken, DirectoryToken

class GenericStructureToken(StructureToken):
    '''
    Abstract class representing a generic item.
    '''
    @abstractmethod
    def __init__(self, name: str):
        super().__init__(name)

    @abstractmethod
    def expand(self) -> list[StructureToken]:
        '''
        Expand the generic structure into a list of structure tokens.
        '''

class GenericDirectoryToken(GenericStructureToken):
    '''
    Generic directory item.
    '''
    def __init__(self, pattern: str,
                 purpose: list[DirectoryPurposeToken] | DirectoryPurposeToken,
                 subtokens: list[StructureToken] | StructureToken):
        self.pattern: str = pattern
        self.purpose: list[DirectoryPurposeToken] = union(purpose)
        self.subtokens: list[StructureToken] = union(subtokens)

    def expand(self) -> list[DirectoryToken]:
        if not self.instantiated:
            raise AssertionError('Structure token is not instantiated!')
        

class GenericFileToken(GenericStructureToken):
    '''
    Generic file item.
    '''
    def __init__(self, pattern: str,
                 purpose: list[FilePurposeToken] | FilePurposeToken):
        self.pattern: str = pattern
        self.purpose: list[FilePurposeToken] = union(purpose)

    def expand(self) -> list[FileToken]:
        pass
