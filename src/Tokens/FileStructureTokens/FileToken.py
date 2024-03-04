'''
FileToken module
'''

from ..PurposeTokens import FilePurposeToken
from .PathToken import PathToken
from ._StructureToken import StructureToken

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
