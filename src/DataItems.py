'''
Represents all possible (required) data items for parsing a dataset.
'''

import os
from typing import Self, Any
from abc import ABC, abstractmethod

from ._utils import union

class _IdentifierToken(ABC):
    '''
    The IdentifierToken class is an abstract class which carries important information into 
    StringFormatTokens for data parsing functions. Subclasses of this class may have specific 
    requirements for content.
    '''
    def __init__(self):
        pass

    @abstractmethod
    def verify_token(self, token: str) -> bool:
        '''
        Checks whether the token is in valid format in accordance with the identifier.
        
        - token: the token to check
        '''

class _StorageToken(_IdentifierToken):
    '''
    The StorageToken class possesses a set of elements which checks upon itself for membership.
    '''
    def __init__(self):
        self.items: set[str] = set()

    def verify_token(self, token: str) -> bool:
        self.items.add(token)
        return True

class _RedundantStorageToken(_StorageToken):
    '''
    The RedundantStorageToken class possesses a set of elements which checks upon itself, but also
    allows for a data item to store a list of items instead of just one value.
    '''

class _UniqueToken(_IdentifierToken):
    '''
    The UniqueToken class possesses a set of elements which checks upon itself for membership.
    '''
    def __init__(self):
        self.items: set[str] = set()

    def verify_token(self, token: str) -> bool:
        self.items.add(token)
        return True

class _WildcardToken(_IdentifierToken):
    '''
    The WildcardToken class represents a generic wildcard which can stand for anything and will not 
    be used for any identifiers.
    '''

    def verify_token(self, token: str) -> bool:
        '''
        Any string passes the wildcard check. Dummy method for assertions.
        
        - token (str): the token parsed from StringFormatToken.match()
        '''
        return True

class _FilenameToken(_UniqueToken):
    '''
    The FilenameToken class is an IdentifierToken which checks for valid filenames.
    '''
    def verify_token(self, token: str) -> bool:
        '''
        Any proper filename passes the check assuming it exists.
        
        - root (str): the root to the main dataset directory.
        - token (str): the token parsed from StringFormatToken.match()
        '''
        return os.path.exists(token) and super().verify_token(token)

class _QuantityToken(_IdentifierToken):
    '''
    Represents a numeric quantity.
    '''
    def verify_token(self, token: Any) -> bool:
        '''
        Passes if token is numeric.
        
        - token (str): the token parsed from StringFormatToken.match()
        '''
        if isinstance(token, (int | float)):
            return True
        elif isinstance(token, str):
            try: float(token)
            except ValueError: return False
            return True
        return False

class _RedundantQuantityToken(_QuantityToken):
    '''
    Represents a numeric quantity.
    '''

class DataType:
    '''
    All possible data types. Container class for IdentifierToken objects with specific purposes.
    
    Instance variables:
    - desc (str): the purpose of the DataType. This should be unique for every new object.
    - storage (bool): whether the DataType stores items contained in it.
    - token_type (type[IdentifierToken]): the token type of the DataType.
    '''

    def __init__(self, desc: str, token_type: _IdentifierToken):
        self.desc: str = desc
        self.token_type: type[_IdentifierToken] = token_type

    def __repr__(self) -> str:
        return f'<{self.desc}>'

    def __eq__(self, other: Self) -> bool:
        if self.__class__ != other.__class__:
            return False
        return self.desc == other.desc

    def verify_token(self, value: str) -> bool:
        '''
        Verify that a given value is valid for the datatype. Calls on internal IdentifierToken
        functions for validation.
        
        - value (str): the value to check if it is compatible with the DataType.
        '''
        return self.token_type.verify_token(value)

class DataTypes:
    '''
    Presets for DataType. These represent valid tokens, and DataType should not be initialized
    directly but rather through these presets.
    '''
    IMAGE_SET = DataType('IMAGE_SET', _RedundantStorageToken())
    IMAGE_SET_ID = DataType('IMAGE_SET_ID', _RedundantStorageToken())
    ABSOLUTE_FILE = DataType('ABSOLUTE_FILE', _FilenameToken())
    RELATIVE_FILE = DataType('RELATIVE_FILE', _FilenameToken())
    IMAGE_NAME = DataType('IMAGE_NAME', _UniqueToken())
    IMAGE_ID = DataType('IMAGE_ID', _UniqueToken())
    CLASS_NAME = DataType('CLASS_NAME', _StorageToken())
    CLASS_ID = DataType('CLASS_ID', _StorageToken())
    BBOX_CLASS_NAME = DataType('BBOX_CLASS_NAME', _RedundantStorageToken())
    BBOX_CLASS_ID = DataType('BBOX_CLASS_ID', _RedundantQuantityToken())
    XMIN = DataType('XMIN', _RedundantQuantityToken())
    YMIN = DataType('YMIN', _RedundantQuantityToken())
    XMAX = DataType('XMAX', _RedundantQuantityToken())
    YMAX = DataType('YMAX', _RedundantQuantityToken())
    WIDTH = DataType('WIDTH', _RedundantQuantityToken())
    HEIGHT = DataType('HEIGHT', _RedundantQuantityToken())
    NAME = DataType('NAME', _WildcardToken())
    GENERIC = DataType('GENERIC', _WildcardToken())

class DataItem:
    '''
    Base, abstract class for representing a data item. Contains a DataType and a value associated
    with it.
    
    Instance variables:
    - delimiter (DataType): the type of the DataItem.
    - value (str): the value associated with the DataType, must be compatible.
    '''
    def __init__(self, delimiter: DataType, value: str):
        assert delimiter.verify_token(value), \
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
    Contains all items required for an entry in the dataset, which contains DataItem objects.
    
    Instance variables:
    - unique (bool): true if this entry contains unique data, paired data otherwise.
    - data (list[DataItem]): list of data items to associate together.
    '''
    def __init__(self, items: list[DataItem] | DataItem):
        items: list[DataItem] = union(items)
        self.unique: bool = any([isinstance(item.delimiter.token_type, _UniqueToken)
                                 for item in items if not isinstance(item, list)])
        self.data: dict[str, DataItem] = {(item.delimiter.desc if not isinstance(item, list) 
                                           else item[0].delimiter.desc): item for item in items}

    @classmethod
    def merge(cls, first: Self, second: Self, overlap=True) -> Self:
        '''
        Merge two data entries together, storing it in this instance.
        
        - other (DataEntry): another data entry to merge into this instance.
        '''
        merged = cls(list(first.data.values()))

        for desc, item in second.data.items():
            if isinstance(item, list) or isinstance(item.delimiter.token_type,
                    (_RedundantQuantityToken, _RedundantStorageToken, _WildcardToken)): continue
            if overlap or isinstance(item.delimiter.token_type, _UniqueToken):
                if desc in merged.data and merged.data[desc] != second.data[desc]: return None

        for desc, item in second.data.items():
            if isinstance(item, list):
                merged.data[desc] = union(merged.data.get(desc, [])) + item
                continue
            if desc not in merged.data:
                merged.data[desc] = item
                continue
            if isinstance(item.delimiter.token_type,
                          (_RedundantStorageToken, _RedundantQuantityToken)):
                merged.data[desc] = union(merged.data.get(desc, [])) + [item]
        return merged

    def merge_inplace(self, other: Self, overlap=True) -> bool:
        for desc, item in other.data.items():
            if isinstance(item, list) or isinstance(item.delimiter.token_type,
                    (_RedundantQuantityToken, _RedundantStorageToken, _WildcardToken)): continue
            if overlap or isinstance(item.delimiter.token_type, _UniqueToken):
                if desc in self.data and self.data[desc] != other.data[desc]: return False

        for desc, item in other.data.items():
            if isinstance(item, list):
                self.data[desc] = union(self.data.get(desc, [])) + item
                continue
            if desc not in self.data:
                self.data[desc] = item
                continue
            if isinstance(item.delimiter.token_type,
                          (_RedundantStorageToken, _RedundantQuantityToken)):
                self.data[desc] = union(self.data.get(desc, [])) + [item]
        return True

    def apply_tokens(self, items: list[DataItem] | DataItem) -> None:
        '''
        Apply new tokens to the item.
        
        - items (list[DataItem] | DataItem): additional items to associate with this data entry.
        '''
        items: list[DataItem] = union(items)
        # execute checks first
        for item in items:
            delim = item.delimiter
            if isinstance(delim.token_type, _UniqueToken):
                assert self.data[delim.desc] == item, \
                       f'Unique identifiers {self.data[delim.desc]} not equal to {item}'
        # merge
        for item in items:
            if item.delimiter.desc not in self.data:
                self.data[item.delimiter.desc] = item
                continue
            if isinstance(item.delimiter.token_type, _RedundantStorageToken):
                self.data[item.delimiter.desc] = union(self.data[item.delimiter.desc])
                self.data[item.delimiter.desc].append(item)

    def get_unique_ids(self) -> list[DataItem]:
        '''
        Return all unique identifier tokens.
        '''
        id_items: list[DataItem] = []
        for item in self.data.values():
            if isinstance(item.delimiter.token_type, _UniqueToken):
                id_items.append(item)
        return id_items

    def __repr__(self) -> str:
        return ' '.join(['DataEntry:']+[str(item) for item in self.data.values()])

def merge_lists(lists: list[list[DataEntry]]) -> list[DataEntry]:
    '''
    Merge two DataEntry lists.
    '''
    if len(lists) == 0: return []
    if len(lists) == 1:
        return lists[0]
    unique_identifiers: list[DataType] = [var for var in vars(DataTypes).values() if
                                            isinstance(var, DataType) and
                                            isinstance(var.token_type, _UniqueToken)]
    hashmaps: dict[str, dict[str, DataEntry]] = {id.desc:{} for id in unique_identifiers}
    for next_list in lists:
        for entry in next_list:
            for identifier in unique_identifiers:
                value = entry.data.get(identifier.desc)
                if not value: continue
                if value.value in hashmaps[identifier.desc]:
                    result = hashmaps[identifier.desc][value.value].merge_inplace(entry)
                    if not result: raise ValueError(f'Found conflicting information when merging \
                        {hashmaps[identifier.desc][value.value]} and {entry}')
                    for id_update in unique_identifiers:
                        value_update = entry.data.get(id_update.desc)
                        if id_update == identifier or not value_update: continue
                        hashmaps[id_update.desc][value_update.value] = hashmaps[identifier.desc][value.value]
                    break
                hashmaps[identifier.desc][value.value] = entry

    data = set()
    for identifier in unique_identifiers:
        data.update(hashmaps[identifier.desc].values())
    return list(data)
