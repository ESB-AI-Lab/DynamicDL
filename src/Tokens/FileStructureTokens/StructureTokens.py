'''
StructureTokens module

StructureToken class: abstract structure class
DirectoryToken class: directory structure class
FileToken class: file structure class
'''

import os
from abc import abstractmethod

from ..._utils import union
from ..Token import Token
from .PathToken import PathToken
from ..PurposeTokens import DirectoryPurposeToken, FilePurposeToken
from ..OtherTokens import StringFormatToken

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

class GenericStructureToken(StructureToken):
    '''
    Abstract class representing a generic item.
    '''
    @abstractmethod
    def __init__(self, pattern: str):
        super().__init__(pattern)

    @abstractmethod
    def expand(self) -> list[StructureToken]:
        '''
        Expand the generic structure into a list of structure tokens.
        '''

    def instantiate(self, path: PathToken) -> None:
        '''
        Instantiates the path of this structure token, and mark as instantiated.
        
        - path: the parent path token.
        '''
        self.path: PathToken = PathToken(path.os_path, path.root)
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
        Instantiate a DirectoryToken.
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
        self.subtokens = instantiate_all(self.subtokens, self.path)

    def __repr__(self) -> str:
        '''
        String of directory object.
        '''
        lines: list[str] = [f'+ Directory ({self.path})',
                 '| - subtokens:']
        cutoff: int = min(len(self.subtokens), 5)
        for subtoken in self.subtokens[:cutoff]:
            lines += ['|   ' + line for line in str(subtoken).split('\n')]
        if len(self.subtokens) > cutoff:
            lines += [f'|   ...{len(self.subtokens) - cutoff} more...']
        return '\n'.join(lines)

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
        
    def __repr__(self) -> str:
        '''
        String of file object.
        '''
        lines = [f'+ File ({self.path})']
        return '\n'.join(lines)

class GenericDirectoryToken(GenericStructureToken):
    '''
    Generic directory item.
    '''
    def __init__(self, pattern: StringFormatToken,
                 purpose: list[DirectoryPurposeToken] | DirectoryPurposeToken,
                 subtokens: list[StructureToken] | StructureToken):
        self.pattern: StringFormatToken = pattern
        self.purpose: list[DirectoryPurposeToken] = union(purpose)
        self.subtokens: list[StructureToken] = union(subtokens)

    def expand(self) -> list[DirectoryToken]:
        if not self.instantiated:
            raise AssertionError('Structure token is not instantiated!')
        all_directories: list[DirectoryToken] = []
        available = get_directories(self.path.get_os_path())
        for directory in available:
            tokens = self.pattern.match(directory)
            if len(tokens) == 0:
                continue
            all_directories.append(DirectoryToken(directory, self.purpose, self.subtokens))
        return all_directories

class GenericFileToken(GenericStructureToken):
    '''
    Generic file item.
    '''
    def __init__(self, pattern: StringFormatToken,
                 purpose: list[FilePurposeToken] | FilePurposeToken):
        self.pattern: StringFormatToken = pattern
        self.purpose: list[FilePurposeToken] = union(purpose)

    def expand(self) -> list[FileToken]:
        if not self.instantiated:
            raise AssertionError('Structure token is not instantiated!')
        all_files: list[FileToken] = []
        available = get_files(self.path.get_os_path())
        for file in available:
            tokens = self.pattern.match(file)
            if len(tokens) == 0:
                continue
            all_files.append(FileToken(file, self.purpose))
        return all_files

def instantiate_all(structures: list[StructureToken], path: PathToken) -> list[StructureToken]:
    '''
    Instantiate a list of structures, including expanding the generic structures.
    Structures is modified in place.
    
    - structures (list[StructureToken]): a list of the structures to instantiate.
    - path (PathToken): the path of the root/parent directory which this is executed from.
    '''
    expanded_structures: list[StructureToken] = []
    for structure in structures:
        if isinstance(structure, GenericStructureToken):
            structure.instantiate(path)
            expanded_structures += structure.expand()

    structures = [structure for structure in structures if not isinstance(structure, GenericStructureToken)]

    # modify in place
    structures += expanded_structures

    for structure in structures:
        structure.instantiate(path)

    return structures

def get_files(os_path: str) -> list[str]:
    '''
    Find all files (not directories) in the given os path.
    '''
    return [name for name in os.listdir(os_path)
            if not os.path.isdir(os.path.join(os_path, name))]

def get_directories(os_path: str) -> list[str]:
    '''
    Find all directories (not files) in the given os path.
    '''
    return [name for name in os.listdir(os_path)
            if os.path.isdir(os.path.join(os_path, name))]
