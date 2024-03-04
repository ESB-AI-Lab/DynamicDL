'''
The fundamental dataset module for importing datasets.
'''

from .FileStructureTokens import PathToken, DirectoryToken, FileToken
from .OtherTokens import ModeToken
from .FormatTokens import FormatToken

from .._utils import union

class Dataset:
    '''
    The Dataset class contains the root of the dataset.
    Instance Variables:
    - modes: list[ModeToken]
        - Contains the type(s) of modes for the training mode that the datasets are fit for.
    - path: PathToken
    - structure: list[DirectoryToken | FileToken]
    - annotation_structure: FormatToken
    '''

    def __init__(self, modes: list[ModeToken] | ModeToken, path: PathToken,
                 structure: list[DirectoryToken | FileToken] | DirectoryToken | FileToken,
                 annotation_structure: FormatToken):
        self.modes = union(modes)
        self.path = path
        self.structure = union(structure)
        self.annotation_structure = annotation_structure
