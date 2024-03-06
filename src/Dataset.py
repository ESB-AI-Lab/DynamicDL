'''
The fundamental dataset module for importing datasets.
'''

from .FormatTokens import FormatToken
from .ModeToken import ModeToken
from .StructureTokens import PathToken, FileToken, StructureToken, DirectoryToken, instantiate_all
from .DataItems import DataEntry, DataTypes, DataType
from .IdentifierTokens import UniqueToken
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
                 structure: list[StructureToken] | StructureToken,
                 annotation_structure: FormatToken):
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
        self.annotation_structure: FormatToken = annotation_structure

        self.structures = instantiate_all(self.structures, self.path, self)

        self.data: list[DataEntry] = self._populate_data(self.structures)

    def _populate_data(self, root: list[StructureToken] | StructureToken) -> list[DataEntry]:
        '''
        Recursively search through root to populate list of data entries in the entire dataset.
        '''
        data: list[DataEntry] = []
        if isinstance(root, list):
            for structure in root:
                data = Dataset._merge_lists(data, self._populate_data(structure))
        elif isinstance(root, DirectoryToken):
            for structure in root.subtokens:
                child_data: list[DataEntry] = self._populate_data(structure)
                for item in child_data:
                    item.apply(structure.data_tokens)
                data = Dataset._merge_lists(data, child_data)
        elif isinstance(root, FileToken):
            data = root.data
        else:
            raise ValueError('Invalid structure! Found StructureToken not of type Dir/File.')
        return data

    @staticmethod
    def _merge_lists(first: list[DataEntry], second: list[DataEntry]) -> list[DataEntry]:
        data: list[DataEntry] = first.copy()
        unique_identifiers: list[DataType] = [var for var in vars(DataTypes).values() if
                                              isinstance(var, DataType) and
                                              isinstance(var.token_type, UniqueToken)]
        # print(unique_identifiers)
        hashmaps: dict[str, dict[str, DataEntry]] = {}
        for identifier in unique_identifiers:
            hashmaps[identifier.desc] = {}
            for entry in data:
                value = entry.data.get(identifier.desc)
                if value:
                    hashmaps[identifier.desc][value.value] = entry

        additional_data: list[DataEntry] = []
        for entry in second:
            merged: bool = False
            for identifier in unique_identifiers:
                value = entry.data.get(identifier.desc)
                if value:
                    if value.value in hashmaps[identifier.desc]:
                        hashmaps[identifier.desc][value.value].merge(entry)
                        merged = True
                        break
            if not merged:
                additional_data.append(entry)
        data += additional_data
        return data

    def __repr__(self) -> str:
        lines = [f'+ Dataset (root {self.path})',
                 f'| - Valid modes: {", ".join([str(mode) for mode in self.modes])}',
                 '| - Files:']
        for structure in self.structures:
            lines += ['|   ' + line for line in str(structure).split('\n')]
        return '\n'.join(lines)

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
                if isinstance(structure, FileToken):
                    return structure
                structure: DirectoryToken
                if structure.path.os_path == path:
                    return structure.subtokens
                curr = structure.subtokens
                index = 0
                continue
            index += 1
        raise ValueError('Path does not exist')
