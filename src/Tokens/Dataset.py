'''
The fundamental dataset module for importing datasets.
'''

from .StructureTokens import PathToken, StructureToken, instantiate_all
from .ModeToken import ModeToken
from . import FormatTokens

from .._utils import union

class Dataset:
    '''
    The Dataset class contains the root of the dataset.
    Instance Variables:
    - modes: list[ModeToken]
        - Contains the type(s) of modes for the training mode that the datasets are fit for.
    - path: PathToken
    - structures: list[StructureToken]
    - annotation_structure: FormatToken
    '''

    def __init__(self, modes: list[ModeToken] | ModeToken, path: PathToken,
                 structure: list[StructureToken] | StructureToken,
                 annotation_structure: FormatTokens):
        '''
        Create a dataset.
        
        - modes (list[ModeToken] | ModeToken): valid training modes for the dataset.
        - path (PathToken): the path to the root directory.
        - structure (list[StructureToken] | StructureToken): the filestructure of the dataset.
        - annotation_structure (FormatToken): the format to parse annotation files in.
        '''
        self.modes: list[ModeToken] = union(modes)
        self.path: PathToken = path
        self.structures: list[StructureToken] = union(structure)
        self.annotation_structure: FormatTokens = annotation_structure

        self.structures = instantiate_all(self.structures, self.path)

        print(self)

    def __repr__(self) -> str:
        lines = [f'+ Dataset (root {self.path})',
                 f'| - Valid modes: {", ".join([str(mode) for mode in self.modes])}',
                 '| - Files:']
        for structure in self.structures:
            lines += ['|   ' + line for line in str(structure).split('\n')]
        return '\n'.join(lines)
