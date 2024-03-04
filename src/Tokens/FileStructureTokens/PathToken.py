'''
PathToken module
'''

import os

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
        self.os_path = os_path
        self.root = root

    def get_os_path(self) -> str:
        '''
        Retrieve the absolute os path to the token structure.
        '''
        return os.path.join(self.root, self.os_path)
