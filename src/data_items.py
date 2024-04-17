'''
Represents all possible (required) data items for parsing a dataset.
'''

import re
import os
from copy import copy
from typing import Any, Union
from typing_extensions import Self

from ._utils import union

class Token:
    '''
    The Token class is an abstract class which carries important information into 
    Data objects for data parsing functions. Subclasses of this class may have specific 
    requirements for content.
    '''
    def __init__(self):
        pass

    def verify_token(self, token: Any) -> bool:
        '''
        Checks whether the token is in valid format in accordance with the identifier.
        
        - token: the token to check
        '''
        return token != ''

    def transform(self, token: Any) -> Any:
        '''
        Transform the token from a string value to token type.
        '''
        return token

class RedundantToken(Token):
    '''
    Allows for redundancy.
    '''
    def transform(self, token: str) -> Any:
        return union(token)

class UniqueToken(Token):
    '''
    The UniqueToken class possesses a set of elements which checks upon itself for membership.
    '''

class WildcardToken(Token):
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

class FilenameToken(UniqueToken):
    '''
    The FilenameToken class is a Token which checks for valid filenames.
    '''
    def verify_token(self, token: str) -> bool:
        '''
        Any proper filename passes the check assuming it exists.
        
        - root (str): the root to the main dataset directory.
        - token (str): the token parsed from StringFormatToken.match()
        '''
        return os.path.exists(token) and super().verify_token(token)

class IDToken(Token):
    '''
    Represents an ID.
    '''
    def verify_token(self, token: Any) -> bool:
        '''
        Passes if token is numeric.
        
        - token (str): the token parsed from StringFormatToken.match()
        '''
        if isinstance(token, (int, float)):
            return True
        elif isinstance(token, str):
            return token.isnumeric()
        return False

    def transform(self, token: str) -> Any:
        return int(token)

class QuantityToken(Token):
    '''
    Represents a numeric quantity.
    '''
    def verify_token(self, token: Any) -> bool:
        '''
        Passes if token is numeric.
        
        - token (str): the token parsed from StringFormatToken.match()
        '''
        if isinstance(token, (int, float)):
            return True
        elif isinstance(token, str):
            try: float(token)
            except ValueError: return False
            return True
        return False

    def transform(self, token: str) -> Any:
        return float(token)

class RedundantQuantityToken(QuantityToken, RedundantToken):
    '''
    Represents a redundant numeric quantity.
    '''

    def transform(self, token: str) -> Any:
        return list(map(lambda x: float(x), union(token)))

class RedundantIDToken(IDToken, RedundantToken):
    '''
    Represents a redundant ID.
    '''

    def transform(self, token: str) -> Any:
        return list(map(lambda x: int(x), union(token)))
    
class RedundantObjectToken(RedundantToken):
    '''
    Represents a segmentation object.
    '''
    def transform(self, token: list) -> list:
        if len(token) > 0 and isinstance(token[0], list): return token
        return [token]

class UniqueIDToken(IDToken, UniqueToken):
    '''
    Represents a unique ID.
    '''

class DataType:
    '''
    All possible data types. Container class for Token objects with specific purposes.
    
    Instance variables:
    - desc (str): the purpose of the DataType. This should be unique for every new object.
    - storage (bool): whether the DataType stores items contained in it.
    - token_type (type[Token]): the token type of the DataType.
    '''

    def __init__(self, desc: str, token_type: Token):
        self.desc: str = desc
        self.token_type: Token = token_type

    def __repr__(self) -> str:
        return f'<{self.desc}>'

    def __eq__(self, other: Self) -> bool:
        if self.__class__ != other.__class__:
            return False
        return self.desc == other.desc
    
    def __hash__(self) -> int:
        return hash(self.desc)

    def verify_token(self, value: str) -> bool:
        '''
        Verify that a given value is valid for the datatype. Calls on internal Token
        functions for validation.
        
        - value (str): the value to check if it is compatible with the DataType.
        '''
        return self.token_type.verify_token(value)

class DataTypes:
    '''
    Presets for DataType. These represent valid tokens, and DataType should not be initialized
    directly but rather through these presets.
    '''
    IMAGE_SET_NAME = DataType('IMAGE_SET_NAME', RedundantToken())
    IMAGE_SET_ID = DataType('IMAGE_SET_ID', RedundantIDToken())
    ABSOLUTE_FILE = DataType('ABSOLUTE_FILE', FilenameToken())
    ABSOLUTE_FILE_SEG = DataType('ABSOLUTE_FILE_SEG', FilenameToken())
    IMAGE_NAME = DataType('IMAGE_NAME', UniqueToken())
    IMAGE_ID = DataType('IMAGE_ID', UniqueIDToken())
    CLASS_NAME = DataType('CLASS_NAME', Token())
    CLASS_ID = DataType('CLASS_ID', IDToken())
    BBOX_CLASS_NAME = DataType('BBOX_CLASS_NAME', RedundantToken())
    BBOX_CLASS_ID = DataType('BBOX_CLASS_ID', RedundantIDToken())
    XMIN = DataType('XMIN', RedundantQuantityToken())
    YMIN = DataType('YMIN', RedundantQuantityToken())
    XMAX = DataType('XMAX', RedundantQuantityToken())
    YMAX = DataType('YMAX', RedundantQuantityToken())
    X1 = DataType('X1', RedundantQuantityToken())
    Y1 = DataType('Y1', RedundantQuantityToken())
    X2 = DataType('X2', RedundantQuantityToken())
    Y2 = DataType('Y2', RedundantQuantityToken())
    WIDTH = DataType('WIDTH', RedundantQuantityToken())
    HEIGHT = DataType('HEIGHT', RedundantQuantityToken())
    SEG_CLASS_NAME = DataType('SEG_CLASS_NAME', RedundantToken())
    SEG_CLASS_ID = DataType('SEG_CLASS_ID', RedundantIDToken())
    X = DataType('X', QuantityToken())
    Y = DataType('Y', QuantityToken())
    NAME = DataType('NAME', WildcardToken())
    GENERIC = DataType('GENERIC', WildcardToken())
    POLYGON = DataType('POLYGON', RedundantObjectToken())

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
        self.value = delimiter.token_type.transform(value)

    def __repr__(self) -> str:
        return f'{self.delimiter}: {self.value}'

    def __eq__(self, other: Self) -> bool:
        if self.__class__ != other.__class__:
            return False
        return self.delimiter == other.delimiter and self.value == other.value

    def add(self, item: Self) -> None:
        '''
        Add an item to current data if it is redundant.
        '''
        assert isinstance(self.delimiter.token_type, RedundantToken), \
            'Cannot add to item which is not redundant'
        if isinstance(self.delimiter.token_type, RedundantObjectToken):
            self.value.append(item.value)
            return
        self.value = self.value + union(item.value)

    @classmethod
    def copy(cls, first: Self) -> Self:
        '''
        Copy self's data.
        '''
        return cls(first.delimiter, copy(first.value))

class DataEntry:
    '''
    Contains all items required for an entry in the dataset, which contains DataItem objects.
    
    Instance variables:
    - unique (bool): true if this entry contains unique data, paired data otherwise.
    - data (list[DataItem]): list of data items to associate together.
    '''
    def __init__(self, items: Union[list[DataItem], DataItem]):
        items: list[DataItem] = union(items)
        self.unique: bool = any([isinstance(item.delimiter.token_type, UniqueToken)
                                 for item in items if not isinstance(item, list)])
        self.data: dict[str, DataItem] = {(item.delimiter.desc if not isinstance(item, list)
                                           else item[0].delimiter.desc): item for item in items}

    @classmethod
    def merge(cls, first: Self, second: Self, overlap: bool = True) -> Self:
        '''
        Merge two data entries together, storing it in a new instance. 
        
        - first (DataEntry): the first data entry to merge.
        - second (DataEntry): the second data entry to merge.
        - overlap (bool): whether to allow nonunique item overlapping. default true.
        
        Returns new DataEntry object.
        '''
        merged = cls(list(first.data.values()))

        for desc, item in second.data.items():
            if isinstance(item.delimiter.token_type, (RedundantToken, WildcardToken)): continue
            if overlap or isinstance(item.delimiter.token_type, UniqueToken):
                if desc in merged.data and merged.data[desc] != second.data[desc]: return None

        for desc, item in second.data.items():
            if desc not in merged.data:
                merged.data[desc] = item
                continue
            if isinstance(item.delimiter.token_type, RedundantToken):
                merged.data[desc].add(item)
        return merged

    def merge_inplace(self, other: Self, overlap=True) -> bool:
        '''
        Merge two data entries together, storing it in this instance. 
        
        - other (DataEntry): the other data entry to merge into this instance.
        - overlap (bool): whether to allow nonunique item overlapping. default true.
        
        Returns true if merge operation succeeded, false otherwise.
        '''
        for desc, item in other.data.items():
            if isinstance(item.delimiter.token_type, (RedundantToken, WildcardToken)): continue
            if overlap or isinstance(item.delimiter.token_type, UniqueToken):
                if desc in self.data and self.data[desc] != other.data[desc]: return False

        for desc, item in other.data.items():
            if desc not in self.data:
                self.data[desc] = item
                continue
            if isinstance(item.delimiter.token_type, RedundantToken):
                # merge_inplace is called only in merge_lists, and we want to preserve when existing
                # redundant values are the same and should not be overwritten/added onto
                if self.data[desc].value == item.value: continue
                self.data[desc].add(item)
        return True

    def apply_tokens(self, items: Union[list[DataItem], DataItem]) -> None:
        '''
        Apply new tokens to the item.
        
        - items (list[DataItem] | DataItem): additional items to associate with this data entry.
        '''
        items: list[DataItem] = [DataItem.copy(item) for item in union(items)]
        # execute checks first
        for item in items:
            if isinstance(item.delimiter.token_type, RedundantToken): continue
            if isinstance(item.delimiter.token_type, UniqueToken):
                assert self.data[item.delimiter.desc] == item, \
                       f'Unique identifiers {self.data[item.delimiter.desc]} not equal to {item}'
        # merge
        for item in items:
            if item.delimiter.desc not in self.data:
                self.data[item.delimiter.desc] = item
            elif isinstance(item.delimiter.token_type, RedundantToken):
                self.data[item.delimiter.desc].add(item)

    def get_unique_ids(self) -> list[DataItem]:
        '''
        Return all unique identifier tokens.
        '''
        id_items: list[DataItem] = []
        for item in self.data.values():
            if isinstance(item.delimiter.token_type, UniqueToken):
                id_items.append(item)
        return id_items

    def __repr__(self) -> str:
        return ' '.join(['DataEntry:']+[str(item) for item in self.data.values()])

class Alias:
    '''
    Class used when a DataType placeholder could be interpreted multiple ways. For example, if
    IMAGE_NAME also contains CLASS_NAME and IMAGE_ID, we can extract all 3 tokens out using
    PatternAlias. Counts for a single wildcard token.
    '''
    def __init__(self, generics: list['Generic']):
        assert len(generics) > 0, 'Must have at least 1 generic in list.'
        self.generics = generics
        self.patterns: list[str] = [generic.name for generic in generics]
        self.aliases: list[tuple[DataType, ...]] = [generic.data for generic in generics]
        self.desc = ''.join([token.desc for alias in self.aliases for token in alias])

    def match(self, entry: str) -> tuple[bool, list[DataItem]]:
        '''
        Return a list of DataItems including all of the possible alias items.
        '''
        result: list[DataItem] = []
        for pattern, alias in zip(self.patterns, self.aliases):
            pattern: str = pattern.replace('{}', '(.+)')
            matches: list[str] = re.findall(pattern, entry)
            try:
                if not matches:
                    return False, []
                # if multiple token matching, extract first matching; else do nothing
                if isinstance(matches[0], tuple):
                    matches = matches[0]
                for data_type, match in zip(alias, matches):
                    result.append(DataItem(data_type, match))
            except AssertionError:
                return False, []
        return True, result

    def substitute(self, values: list[DataItem]) -> str:
        '''
        Given the list (in order) of items, substitute the generic to retrieve original data string.
        '''
        return Generic(self.patterns[0], *self.aliases[0]).substitute(values[:len(self.aliases[0])])

    def length(self) -> int:
        '''
        Get the length of the alias, i.e. how many tokens there are.
        '''
        return len(self.patterns)

    def __repr__(self) -> str:
        return str(dict(zip(self.patterns, self.aliases)))

class Static:
    '''
    Represents an object with a static name. Can contain data.
    '''
    def __init__(self, name: str, data: Union[list[DataItem], DataItem] = []):
        self.name: str = name
        self.data: list[DataItem] = union(data)

    def match(self, entry: str) -> tuple[bool, list[DataItem]]:
        '''
        Checks if the entry string matches this static item.
        '''
        matched: bool = entry == self.name
        data: list[DataItem] = self.data if matched else []
        return matched, data

    def __repr__(self) -> str:
        return f'*-{self.name} ({", ".join([str(item) for item in self.data])})-*'

class Generic:
    '''
    Represents an object with a generic name.
    '''
    def __init__(self, name: str, *data: Union[DataType, Alias], ignore: Union[list[str], str] = []):
        if isinstance(name, DataType):
            data = tuple([name])
            name = '{}'
        assert len(data) == name.count('{}'), 'Format must have same number of wildcards'
        self.name: str = name
        self.data: tuple[Union[DataType, Alias], ...] = data
        self.ignore: list[str] = union(ignore)

    def match(self, entry: str) -> tuple[bool, list[DataItem]]:
        '''
        Return a list of the tokens' string values provided an entry string which follows the 
        pattern.
        
        - entry (str): the string to match to the pattern, assuming it does match
        '''
        for ignore_pattern in self.ignore:
            ignore_pattern = '^' + self.name.replace('{}', '(.+)') + '+$'
            if re.findall(ignore_pattern, entry): return False, []
        pattern: str = '^' + self.name.replace('{}', '(.+)') + '+$'
        matches: list[str] = re.findall(pattern, entry)
        result: list[DataItem] = []

        if not matches:
            return False, []
        # if multiple token matching, extract first matching; else do nothing
        try:
            if isinstance(matches[0], tuple): matches = matches[0]
            for data_type, match in zip(self.data, matches):
                if not isinstance(data_type, Alias):
                    result.append(DataItem(data_type, match))
                    continue
                success, matched = data_type.match(match)
                if not success: return False, []
                result += matched
        except AssertionError: return False, []
        return True, result

    def substitute(self, values: Union[list[DataItem], DataItem]) -> str:
        '''
        Return the string representation of the values provided string representations for each
        token as a list
        
        - values (list[str] | str): the values of the tokens to replace, in order
        '''
        values: list[DataItem] = union(values)
        substitutions: list[str] = []
        index: int = 0
        for token in self.data:
            if isinstance(token, Alias):
                substitutions.append(token.substitute(values[index:]))
                index += token.length()
            else:
                if isinstance(values[index].delimiter.token_type, RedundantToken):
                    assert len(values[index].value) == 1, \
                        'Redundant token cannot have multiple values in a generic'
                    substitutions.append(values[index].value[0])
                else:
                    substitutions.append(values[index].value)
                index += 1
        return self.name.format(*substitutions)

    def __repr__(self) -> str:
        return f'{self.name} | {self.data}'

class Folder(Generic):
    '''
    Generic for directories only.
    '''

class File(Generic):
    '''
    Generic for files only.
    '''

class Image:
    '''
    Generic image.
    '''
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "Image"

class SegmentationImage:
    '''
    Segmentation mapped image.
    '''
    def __init__(self):
        pass
    
    def __repr__(self) -> str:
        return "Segmentation Image"