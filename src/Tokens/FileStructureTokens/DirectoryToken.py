'''
DirectoryToken module
'''

from ..PurposeTokens import DirectoryPurposeToken
from .PathToken import PathToken
from ._StructureToken import StructureToken

from ..._utils import union

class DirectoryToken(StructureToken):
    '''
    Represents a directory in the file structure.
    Instance Variables:
    - path: PathToken
    - purpose: list[DirectoryPurposeToken]
    - subtokens: list[StructureToken]
    '''
    def __init__(self, path: PathToken,
                 purpose: list[DirectoryPurposeToken] | DirectoryPurposeToken):
        '''
        Instantiate a FileToken.
        '''
        self.purpose = union(purpose)
        self.subtokens = []
        super().__init__(path)
