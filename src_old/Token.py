'''
Abstract base class for all Tokens.
'''

import os
from typing import Self
from abc import ABC, abstractmethod

class Token(ABC):
    '''
    The Token class is the base class for anything rooted in the configuration. Many different types
    of Tokens extend it, and the extensions will have present types.
    '''
    @abstractmethod
    def __init__(self):
        pass

class FormatToken(Token):
    '''
    The FormatToken abstract class is a framework for format classes to support utility functions
    for parsing annotation data.
    '''

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_data(self, file: str) -> tuple[list, list]:
        '''
        Retrieve data.
        
        - file (str): path to the annotation file.
        '''

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