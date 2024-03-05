'''
Module for all StructureToken types

PathToken class: path representation class
StructureToken class: abstract structure class
GenericStructureToken class: abstract generic structure class
DirectoryToken class: directory structure class
FileToken class: file structure class
GenericDirectoryToken class: generic directory structure class
GenericFileToken class: generic file structure class
'''

import os
from typing import Self
from abc import abstractmethod

from .._utils import union
from . import Token, DirectoryPurposeToken, FilePurposeToken, PurposeToken, StringFormatToken, \
              File, DataItem

class PathToken(Token):
    '''
    The PathToken class represents an OS path, and also can be used to traverse the token structure.
    
    Instance Variables:
    - os_path (str): The relative OS path, starting from the root
    - root (str): The absolute OS path to the root of the dataset
    '''

    def __init__(self, os_path: str, root: str):
        '''
        Initialize the PathToken.
        '''
        assert os.path.exists(os.path.join(root, os_path)), 'Path must be valid'
        self.os_path: str = os_path
        self.root: str = root

    def get_os_path(self) -> str:
        '''
        Retrieve the absolute os path to the token structure.
        '''
        return os.path.join(self.root, self.os_path)

    def subpath(self, name):
        '''
        Create a PathToken that is a subpath of the provided token.
        '''
        return PathToken(os.path.join(self.os_path, name), self.root)

    @classmethod
    def init_root(cls, base_dir: str) -> Self:
        '''
        Initialize the root PathToken.
        
        - base_dir (str): the base directory of the dataset.
        '''
        return cls('', base_dir)

    def __repr__(self) -> str:
        return self.get_os_path()
    
    def __eq__(self, other: Self) -> bool:
        if self.__class__ != other.__class__:
            return False
        return self.os_path == other.os_path and self.root == other.root

    def get_files(self) -> list[str]:
        '''
        Find all files (not directories) in this path. This token must be a directory type.
        '''
        assert os.path.isdir(self.get_os_path()), 'This PathToken does not represent a directory!'
        return [name for name in os.listdir(self.get_os_path())
                if not os.path.isdir(os.path.join(self.get_os_path(), name))]

    def get_directories(self) -> list[str]:
        '''
        Find all directories (not files) in the given os path.
        '''
        assert os.path.isdir(self.get_os_path()), 'This PathToken does not represent a directory!'
        return [name for name in os.listdir(self.get_os_path())
                if os.path.isdir(os.path.join(self.get_os_path(), name))]

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

    def instantiate(self, path: PathToken, dataset) -> None:
        '''
        Instantiates the path of this structure token, and mark as instantiated.
        
        - path (PathToken): the parent path token.
        - dataset (Dataset): the dataset this structure token belongs to.
        '''
        self.path: PathToken = path.subpath(self.name)
        self.dataset = dataset
        self.instantiated = True

class GenericStructureToken(StructureToken):
    '''
    Abstract class representing a generic item.
    '''
    @abstractmethod
    def __init__(self, pattern: str,
                 purpose: PurposeToken):
        self.pattern: StringFormatToken = pattern
        self.purpose: PurposeToken = purpose
        super().__init__(str(pattern))

    @abstractmethod
    def expand(self) -> list[StructureToken]:
        '''
        Expand the generic structure into a list of structure tokens.
        '''

    def instantiate(self, path: PathToken, dataset) -> None:
        '''
        Instantiates the path of this structure token, and mark as instantiated.
        
        - path: the parent path token.
        - dataset: the dataset this structure token belongs to.
        '''
        self.path: PathToken = PathToken(path.os_path, path.root)
        self.dataset: dataset
        self.instantiated = True

class DirectoryToken(StructureToken):
    '''
    Represents a directory in the file structure.
    
    Instance Variables:
    - name (str): the local name of the directory.
    - path (PathToken): the path token for this directory.
    - purpose (list[DirectoryPurposeToken] | DirectoryPurposeToken): the purpose of the directory.
    - subtokens (list[StructureToken] | GenericStructureToken): the items contained in the dir.
    '''
    def __init__(self, name: str,
                 purpose: list[DirectoryPurposeToken] | DirectoryPurposeToken,
                 subtokens: list[StructureToken] | GenericStructureToken):
        '''
        Instantiate a DirectoryToken.
        
        - name (str): the local name of the directory.
        - purpose (list[DirectoryPurposeToken] | DirectoryPurposeToken): the directory's purpose
        - subtokens (list[StructureToken] | GenericStructureToken): the items contained in the dir.
        '''
        self.purpose: list[DirectoryPurposeToken] = union(purpose)
        self.subtokens: list[StructureToken] = union(subtokens)
        super().__init__(name)

    def instantiate(self, path: PathToken, dataset) -> None:
        '''
        Instantiates the path of this directory token, and mark as instantiated.
        
        - path (PathToken): the parent path token.
        '''
        super().instantiate(path, dataset)
        self.subtokens = instantiate_all(self.subtokens, self.path, dataset)

    def __repr__(self) -> str:
        lines: list[str] = [f'+ Directory ({self.path})',
                 f'| - Purpose: {self.purpose}',
                 '| - Contents:']
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
    - path (PathToken): the path token for this file.
    - purpose (FilePurposeToken): the purpose token for the file.
    - format (FormatToken): the format of this file.
    '''
    def __init__(self, path: PathToken, purpose: FilePurposeToken):
        '''
        Instantiate a FileToken.
        
        - path (PathToken): the parent path token.
        - purpose (FilePurposeToken): the purpose token for the file.
        '''
        self.purpose: FilePurposeToken = purpose
        self.data: list[list[DataItem]] = None
        super().__init__(path)

    def instantiate(self, path: PathToken, dataset) -> bool:
        super().instantiate(path, dataset)
        if self.purpose == File.ANNOTATION:
            self.data: list[list[DataItem]] = dataset.annotation_structure.get_data(self.path)

    def __repr__(self) -> str:
        lines = [f'+ File ({self.path})',
                 f'| - Purpose: {self.purpose}']
        return '\n'.join(lines)

class GenericDirectoryToken(GenericStructureToken):
    '''
    Generic directory item.
    
    Instance variables:
    - name (StringFormatToken): string representation of pattern, not meant to be used.
    - pattern (StringFormatToken): the name format that the directory name must obey.
    - purpose (list[DirectoryPurposeToken] | DirectoryPurposeToken): the purpose of the directory.
    - subtokens (list[StructureToken] | GenericStructureToken): the items contained in the dir.
    '''
    def __init__(self, pattern: StringFormatToken,
                 purpose: DirectoryPurposeToken,
                 subtokens: list[StructureToken] | StructureToken):
        self.subtokens: list[StructureToken] = union(subtokens)
        super().__init__(pattern, purpose)

    def expand(self) -> list[DirectoryToken]:
        if not self.instantiated:
            raise AssertionError('Structure token is not instantiated!')
        all_directories: list[DirectoryToken] = []
        available = self.path.get_directories()
        for directory in available:
            tokens = self.pattern.match(directory, insertion=True)
            if len(tokens) == 0:
                continue
            all_directories.append(DirectoryToken(directory, self.purpose, self.subtokens))
        return all_directories

class GenericFileToken(GenericStructureToken):
    '''
    Generic file item.
    
    Instance variables:
    - name (StringFormatToken): string representation of pattern, not meant to be used.
    - pattern (StringFormatToken): the name format that the directory name must obey.
    - purpose (list[FilePurposeToken] | FilePurposeToken): the purpose of the directory.
    '''
    def __init__(self, pattern: StringFormatToken,
                 purpose: FilePurposeToken):
        super().__init__(pattern, purpose)

    def expand(self) -> list[FileToken]:
        if not self.instantiated:
            raise AssertionError('Structure token is not instantiated!')
        all_files: list[FileToken] = []
        available = self.path.get_files()
        for file in available:
            tokens = self.pattern.match(file, insertion=True)
            if len(tokens) == 0:
                continue
            all_files.append(FileToken(file, self.purpose))
        return all_files

def instantiate_all(structures: list[StructureToken], path: PathToken,
                    dataset) -> list[StructureToken]:
    '''
    Instantiate a list of structures, including expanding the generic structures.
    Structures is modified in place.
    
    - structures (list[StructureToken]): a list of the structures to instantiate.
    - path (PathToken): the path of the root/parent directory which this is executed from.
    '''
    expanded_structures: list[StructureToken] = []
    for structure in structures:
        if isinstance(structure, GenericStructureToken):
            structure.instantiate(path, dataset)
            expanded_structures += structure.expand()

    structures = [structure for structure in structures \
                  if not isinstance(structure, GenericStructureToken)]

    structures = structures + expanded_structures

    for structure in structures:
        structure.instantiate(path, dataset)

    return structures
