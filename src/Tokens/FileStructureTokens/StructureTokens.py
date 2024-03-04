'''
StructureTokens module

StructureToken class: abstract structure class
DirectoryToken class: directory structure class
FileToken class: file structure class
'''

from abc import abstractmethod

from ..._utils import union
from ..Token import Token
from .PathToken import PathToken
from ..PurposeTokens import DirectoryPurposeToken, FilePurposeToken
from .GenericTokens import GenericStructureToken
from .utils import instantiate_all

class StructureToken(Token):
    '''
    Abstract class representing either a file or a directory
    
    Instance Variables:
    - name (str): The filename of the item.
    - instantiated (bool): Whether it has been instantiated yet. Called from
                           the Dataset class, and can only be active if True.
    '''

    @abstractmethod
    def __init__(self, name: str):
        self.name: str = name
        self.instantiated: bool = False

    def instantiate(self, path: PathToken) -> None:
        '''
        Instantiates the path of this structure token, and mark as instantiated.
        
        - path: the parent path token.
        '''
        self.path: PathToken = path.subpath(self.name)
        self.instantiated = True

class DirectoryToken(StructureToken):
    '''
    Represents a directory in the file structure.
    Instance Variables:
    - path: PathToken
    - purpose: list[DirectoryPurposeToken]
    - subtokens: list[StructureToken] | GenericStructureToken
    '''
    def __init__(self, name: str,
                 purpose: list[DirectoryPurposeToken] | DirectoryPurposeToken,
                 subtokens: list[StructureToken] | GenericStructureToken):
        '''
        Instantiate a FileToken.
        '''
        self.purpose: list[DirectoryPurposeToken] = union(purpose)
        self.subtokens: list[StructureToken] = union(subtokens)
        super().__init__(name)

    def instantiate(self, path: PathToken) -> None:
        '''
        Instantiates the path of this directory token, and mark as instantiated.
        
        - path: the parent path token.
        '''
        super().instantiate(path)
        instantiate_all(self.subtokens)

class FileToken(StructureToken):
    '''
    Represents a file in the file structure.
    Instance Variables:
    - path: PathToken
    - purpose: FilePurposeToken
    - format: FormatToken
    '''
    def __init__(self, path: PathToken, purpose: FilePurposeToken):
        '''
        Instantiate a FileToken.
        '''
        self.purpose = purpose
        super().__init__(path)
