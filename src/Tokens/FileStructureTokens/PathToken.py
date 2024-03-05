'''
PathToken module
'''

import os
from typing import Self

from ..Token import Token

class PathToken(Token):
    '''
    The PathToken class represents an OS path, and also can be used to traverse the token structure.
    
    Instance Variables:
    - os_path: str
        - The relative OS path, starting from the root
    - root: str
        - The absolute OS path to the root of the dataset
    Methods:
    - get_os_path(self) -> str
        - Retrieve the absolute os path to the token structure.
    '''

    def __init__(self, os_path: str, root: str):
        '''
        Initialize the PathToken.
        '''
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
