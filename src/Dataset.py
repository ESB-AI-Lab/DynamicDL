'''
The fundamental dataset module for importing datasets.
'''

from .ModeToken import ModeToken
from .Token import PathToken
from .StructureTokens import FileToken, StructureToken, DirectoryToken, instantiate_all
from .DataItems import DataEntry
from .FormatTokens import merge_lists
from ._utils import union

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
                 structure: list[StructureToken] | StructureToken):
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

        self.structures = instantiate_all(self.structures, self.path)
        self.data, self.pairing_data = self._populate_data(self.structures)

        for pair in self.pairing_data:
            pair.apply_pairing(self.data)

        for entry in self.data:
            if 'GENERIC' in entry.data:
                entry.data.pop('GENERIC')

    def _populate_data(self, root: list[StructureToken] | StructureToken) -> \
            tuple[list[DataEntry], list[DataEntry]]:
        '''
        Recursively search through root to populate list of data entries in the entire dataset.
        
        - root (list[StructureToken] | StructureToken): the sublocation to begin searching at.
        '''
        data: list[DataEntry] = []
        pairing_data: list[DataEntry] = []
        if isinstance(root, list):
            for structure in root:
                child_data, child_pairing_data = self._populate_data(structure)
                data = merge_lists(data, child_data)
                pairing_data += child_pairing_data
        elif isinstance(root, DirectoryToken):
            for structure in root.subtokens:
                child_data, child_pairing_data = self._populate_data(structure)
                pairing_data += child_pairing_data
                for item in child_data:
                    if item.unique:
                        item.apply_tokens(structure.data_tokens)
                    else:
                        pairing_data.append(item)
                data = merge_lists(data, child_data)
        elif isinstance(root, FileToken): return root.data, root.pairing_data
        else:
            raise ValueError('Invalid structure! Found StructureToken not of type Dir/File.')
        return data, pairing_data

    def get(self, path: str) -> FileToken | list[StructureToken]:
        '''
        Get the structure token object at a given path.
        '''
        if path == self.path:
            return self.structures
        curr: list[StructureToken] = self.structures
        index = 0
        while index < len(curr):
            structure = curr[index]
            if structure.path.os_path in path:
                if isinstance(structure, FileToken): return structure
                structure: DirectoryToken
                if structure.path.os_path == path: return structure.subtokens
                curr = structure.subtokens
                index = 0
                continue
            index += 1
        raise ValueError('Path does not exist')

    def __repr__(self) -> str:
        lines = [f'+ Dataset (root {self.path})',
                 f'| - Valid modes: {", ".join([str(mode) for mode in self.modes])}',
                 '| - Files:']
        for structure in self.structures:
            lines += ['|   ' + line for line in str(structure).split('\n')]
        return '\n'.join(lines)
