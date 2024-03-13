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

from abc import abstractmethod

from ._utils import union
from .DataItems import DataEntry, DataItem, DataTypes
from .Token import Token, PathToken, FormatToken
from .PurposeTokens import DirectoryPurposeToken, FilePurposeToken, PurposeToken, File
from .FormatTokens import StringFormatToken



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
        
        - path (PathToken): the parent path token.
        - dataset (Dataset): the dataset this structure token belongs to.
        '''
        self.path: PathToken = path.subpath(self.name)
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

    def instantiate(self, path: PathToken) -> None:
        '''
        Instantiates the path of this structure token, and mark as instantiated.
        
        - path: the parent path token.
        - dataset: the dataset this structure token belongs to.
        '''
        self.path: PathToken = PathToken(path.os_path, path.root)
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
                 subtokens: list[StructureToken] | GenericStructureToken,
                 data_tokens: list[DataItem] = []):
        '''
        Instantiate a DirectoryToken.
        
        - name (str): the local name of the directory.
        - purpose (list[DirectoryPurposeToken] | DirectoryPurposeToken): the directory's purpose
        - subtokens (list[StructureToken] | GenericStructureToken): the items contained in the dir.
        '''
        self.purpose: list[DirectoryPurposeToken] = union(purpose)
        self.subtokens: list[StructureToken] = union(subtokens)
        self.data_tokens: list[DataItem] = data_tokens
        super().__init__(name)

    def instantiate(self, path: PathToken) -> None:
        '''
        Instantiates the path of this directory token, and mark as instantiated.
        
        - path (PathToken): the parent path token.
        '''
        super().instantiate(path)
        self.subtokens = instantiate_all(self.subtokens, self.path)

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
    def __init__(self, name: str, purpose: FilePurposeToken,
                 data_tokens: list[DataItem] = [], format_token: FormatToken = None):
        '''
        Instantiate a FileToken.
        
        - name (str): the name of the file.
        - purpose (FilePurposeToken): the purpose token for the file.
        '''
        self.purpose: FilePurposeToken = purpose
        self.data: list[DataEntry] = None
        self.data_tokens: list[DataItem] = data_tokens
        if self.purpose == File.ANNOTATION:
            assert format_token is not None, 'Must have a format for annotation files.'
        self.format_token: FormatToken = format_token
        super().__init__(name)

    def instantiate(self, path: PathToken) -> bool:
        super().instantiate(path)
        if self.purpose == File.ANNOTATION:
            self.data, self.pairing_data = self.format_token.get_data(self.path)
        if self.purpose == File.IMAGE:
            self.data_tokens += [DataItem(DataTypes.ABSOLUTE_FILE, self.path.get_os_path())]
            self.data: list[DataEntry] = [DataEntry(self.data_tokens)]
            self.pairing_data: list[DataEntry] = []

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
            tokens = self.pattern.match(directory)
            if len(tokens) == 0:
                continue
            all_directories.append(DirectoryToken(directory, self.purpose, self.subtokens,
                                                  data_tokens=tokens))
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
                 purpose: FilePurposeToken, format_token: FormatToken = None):
        if purpose == File.ANNOTATION:
            assert format_token is not None, 'Must have a format for annotation files.'
        self.format_token: FormatToken = format_token
        super().__init__(pattern, purpose)

    def expand(self) -> list[FileToken]:
        if not self.instantiated:
            raise AssertionError('Structure token is not instantiated!')
        all_files: list[FileToken] = []
        available = self.path.get_files()
        for file in available:
            tokens = self.pattern.match(file)
            if len(tokens) == 0:
                continue
            all_files.append(FileToken(file, self.purpose, data_tokens=tokens,
                                       format_token=self.format_token))
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

    structures = [structure for structure in structures \
                  if not isinstance(structure, GenericStructureToken)]

    structures = structures + expanded_structures

    for structure in structures:
        structure.instantiate(path)

    return structures
