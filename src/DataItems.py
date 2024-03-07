'''
Represents all possible (required) data items for parsing a dataset.
'''

from typing import Self

from .IdentifierTokens import IdentifierToken, StorageToken, WildcardToken, FilenameToken, \
                              QuantityToken, UniqueToken, RedundantStorageToken
from ._utils import union

class DataType:
    '''
    All possible data types.
    '''
    def __init__(self, desc: str, token_type: IdentifierToken):
        self.desc: str = desc
        self.storage: bool = False
        if isinstance(token_type, (StorageToken, UniqueToken)):
            self.storage = True
        self.token_type: type[IdentifierToken] = token_type

    def __repr__(self) -> str:
        return f'<{self.desc}>'

    def __eq__(self, other: Self) -> bool:
        if self.__class__ != other.__class__:
            return False
        return self.desc == other.desc

    def verify_token(self, value: str, insertion: bool = False) -> bool:
        '''
        Verify that a given value is valid for the datatype.
        '''
        if self.storage:
            return self.token_type.verify_token(value, insertion=insertion)
        return self.token_type.verify_token(value)

class DataTypes:
    '''
    Presets for DataType.
    '''
    IMAGE_SET: DataType = DataType('IMAGE_SET', RedundantStorageToken())
    IMAGE_SET_ID: DataType = DataType('IMAGE_SET_ID', RedundantStorageToken())
    ABSOLUTE_FILE: DataType = DataType('ABSOLUTE_FILE', FilenameToken())
    RELATIVE_FILE: DataType = DataType('RELATIVE_FILE', FilenameToken())
    IMAGE_NAME: DataType = DataType('IMAGE_NAME', UniqueToken())
    IMAGE_ID: DataType = DataType('IMAGE_ID', UniqueToken())
    CLASS_NAME: DataType = DataType('CLASS_NAME', StorageToken())
    CLASS_ID: DataType = DataType('CLASS_ID', StorageToken())
    XMIN: DataType = DataType('XMIN', QuantityToken())
    YMIN: DataType = DataType('YMIN', QuantityToken())
    XMAX: DataType = DataType('XMAX', QuantityToken())
    YMAX: DataType = DataType('YMAX', QuantityToken())
    WIDTH: DataType = DataType('WIDTH', QuantityToken())
    HEIGHT: DataType = DataType('HEIGHT', QuantityToken())
    GENERIC: DataType = DataType('GENERIC', WildcardToken())

class DataItem:
    '''
    Base, abstract class for representing a data item.
    '''
    def __init__(self, delimiter: DataType, value: str, insertion=False):
        assert delimiter.verify_token(value, insertion = insertion), \
               f'Value {value} is invalid for given delimiter type {delimiter}'
        self.delimiter: DataType = delimiter
        self.value: str = value

    def __repr__(self) -> str:
        return f'{self.delimiter}: {self.value}'

    def __eq__(self, other: Self) -> bool:
        if self.__class__ != other.__class__:
            return False
        return self.delimiter == other.delimiter and self.value == other.value

class DataEntry:
    '''
    Contains all items required for an entry in the dataset.
    
    Instance variables:
    - unique (bool): true if this entry contains unique data, paired data otherwise.
    - data (list[DataItem] | DataItem): list of data items to associate together.
    '''
    def __init__(self, items: list[DataItem] | DataItem):
        items: list[DataItem] = union(items)
        self.unique: bool = any([isinstance(item.delimiter.token_type, UniqueToken)
                                 for item in items])
        self.data: dict[str, DataItem] = {item.delimiter.desc: item for item in items}

    def merge(self, other: Self) -> None:
        '''
        Merge two data entries together, storing it in this instance.
        
        - other (DataEntry): another data entry to merge into this instance.
        '''
        # execute checks first
        for desc, item in other.data.items():
            if isinstance(item.delimiter.token_type, UniqueToken):
                assert desc not in self.data or self.data[desc] == other.data[desc], \
                       f'Unique identifiers {self.data[desc]} not equal to {other.data[desc]}'
        # merge
        for desc, item in other.data.items():
            if desc not in self.data:
                self.data[desc] = item
                continue
            if isinstance(item.delimiter.token_type, RedundantStorageToken):
                self.data[desc] = union(self.data[desc])
                self.data[desc].append(item)

    def apply_tokens(self, items: list[DataItem] | DataItem) -> None:
        '''
        Apply new tokens to the item.
        
        - items (list[DataItem] | DataItem): additional items to associate with this data entry.
        '''
        items: list[DataItem] = union(items)
        # execute checks first
        for item in items:
            delim = item.delimiter
            if isinstance(delim.token_type, UniqueToken):
                assert self.data[delim.desc] == item, \
                       f'Unique identifiers {self.data[delim.desc]} not equal to {item}'
        # merge
        for item in items:
            if item.delimiter.desc not in self.data:
                self.data[item.delimiter.desc] = item
                continue
            if isinstance(item.delimiter.token_type, RedundantStorageToken):
                self.data[item.delimiter.desc] = union(self.data[item.delimiter.desc])
                self.data[item.delimiter.desc].append(item)

    def apply_pairing(self, entries: list[Self] | Self) -> None:
        '''
        If entry is a non-unique entry, apply its pairing to all entries to populate data.
        '''
        if self.unique:
            raise ValueError('Can only apply pairing for non-unique DataEntry instances.')
        entries: list[Self] = union(entries)
        for entry in entries:
            if any([item in entry.data.values() for item in self.data.values()]):
                entry.apply_tokens(list(self.data.values()))

    def get_unique_ids(self) -> list[DataItem]:
        '''
        Return identifier tokens.
        '''
        id_items: list[DataItem] = []
        for item in self.data.values():
            if isinstance(item.delimiter.token_type, UniqueToken):
                id_items.append(item)
        return id_items

    def __repr__(self) -> str:
        return '\n'.join(['DataEntry:']+[str(item) for item in self.data.values()])
