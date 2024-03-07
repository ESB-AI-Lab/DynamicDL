'''
The fundamental dataset module for importing datasets.
'''

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

    def _populate_data(self, root: list[StructureToken] | StructureToken) -> \
            tuple[list[DataEntry], list[DataEntry]]:
        '''
        Recursively search through root to populate list of data entries in the entire dataset.
        '''
        data: list[DataEntry] = []
        pairing_data: list[DataEntry] = []
        if isinstance(root, list):
            for structure in root:
                child_data, child_pairing_data = self._populate_data(structure)
                data = Dataset._merge_lists(data, child_data)
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
                data = Dataset._merge_lists(data, child_data)
        elif isinstance(root, FileToken):
            for entry in root.data:
                if entry.unique:
                    data.append(entry)
                else:
                    pairing_data.append(entry)
        else:
            raise ValueError('Invalid structure! Found StructureToken not of type Dir/File.')
        return data, pairing_data

    @staticmethod
    def _merge_lists(first: list[DataEntry], second: list[DataEntry]) -> list[DataEntry]:
        data: list[DataEntry] = first.copy()
        unique_identifiers: list[DataType] = [var for var in vars(DataTypes).values() if
                                              isinstance(var, DataType) and
                                              isinstance(var.token_type, UniqueToken)]
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
