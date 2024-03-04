'''
StructureToken module
'''

from abc import abstractmethod

from ..Token import Token
from .PathToken import PathToken

class StructureToken(Token):
    '''
    Abstract class representing either a file or a directory
    
    Instance Variables:
    - path: PathToken
        - Contains the path token to the directory.
    '''

    @abstractmethod
    def __init__(self, path: PathToken):
        self.path = path
