'''
Module `data_items`

Module with main data items for dataset parsing.

Token (pseudo-private) classes:
 - `Token`
 - `RedundantToken`
 - `RedundantQuantityToken`
 - `RedundantIDToken`
 - `RedundantObjectToken`
 - `UniqueToken`
 - `FilenameToken`
 - `UniqueIDToken`
 - `WildcardToken`
 - `IDToken`
 - `QuantityToken`

Other classes:
 - `DataType` (data types, psuedo-private)
 - `DataTypes` (presets for data types)
 - `DataItem` (data items containing values of specific data types)
 - `DataEntry` (combinations of associated data items)
 - `Alias` (combination of data types in same generic pattern)
 - `Static` (represents one item, possibly with data)
 - `Generic` (represents a generic item, with data)
 - `Folder` (folder/directory generic)
 - `File` (file generic)
 - `ImageFile` (image file generic)
 - `Image` (image file token)
 - `SegmentationImage` (segmentation image file token)
'''

import re
import os
from copy import copy
from typing import Any, Union, Optional
from typing_extensions import Self

from ._utils import union, Warnings

__all__ = [
    'DataTypes',
    'DataItem',
    'DataEntry',
    'Alias',
    'Static',
    'Generic',
    'Folder',
    'File',
    'ImageFile',
    'Image',
    'SegmentationImage'
]

class Token:
    '''
    The Token class is the base class which carries important information into Data objects for data
    parsing functions. Subclasses of this class may have specific requirements for content.
    
    All implementations of the Token class should not be static but also not use self, for
    compatibility reasons (may be changed in the future)
    '''
    def __init__(self) -> None:
        pass

    def verify_token(self, token: Any) -> bool:
        '''
        Checks whether the token is in valid format in accordance with the identifier. 
        
         - `token`: the token to check
        '''
        return token != ''

    def transform(self, token: Any) -> Any:
        '''
        Transform the token from a string value to token type.
        
         - `token`: the token to transform
        '''
        return token

class RedundantToken(Token):
    '''
    RedundantToken items are used for when a data item stores multiple values of itself per image
    or unique item. Cases like these include multiple bounding boxes or segmentation objects.
    '''
    def transform(self, token: str) -> Any:
        return union(token)

class UniqueToken(Token):
    '''
    UniqueToken items are used when an identifier is a unique item pertaining to any property of an
    image or entry. Unique tokens serve as valid IDs for identifying each data entry in the dataset.
    '''

class WildcardToken(Token):
    '''
    The WildcardToken class represents a generic wildcard which can stand for anything and will not 
    be used for any identifiers. The key difference is that these tokens are not affected by merge
    operations.
    '''

class FilenameToken(UniqueToken):
    '''
    The FilenameToken class is a Token which checks for valid absolute filenames.
    '''
    def verify_token(self, token: Any) -> bool:
        return super().verify_token(token) and os.path.exists(token)

class IDToken(Token):
    '''
    Represents an ID. Items must be integers.
    '''
    def verify_token(self, token: Any) -> bool:
        return isinstance(token, int)

    def transform(self, token: str) -> Any:
        return int(token)

class QuantityToken(Token):
    '''
    Represents a numeric quantity. Can be int or float.
    '''
    def verify_token(self, token: Any) -> bool:
        return isinstance(token, (int, float))

    def transform(self, token: str) -> Any:
        return float(token)

class RedundantQuantityToken(QuantityToken, RedundantToken):
    '''
    Represents a redundant numeric quantity.
    '''
    def verify_token(self, token: Any) -> bool:
        return isinstance(token, list) and all(isinstance(x, (float, int)) for x in token)

    def transform(self, token: str) -> Any:
        return list(map(float, union(token)))

class RedundantIDToken(IDToken, RedundantToken):
    '''
    Represents a redundant ID.
    '''
    def verify_token(self, token: Any) -> bool:
        return isinstance(token, list) and all(isinstance(x, int) for x in token)

    def transform(self, token: str) -> Any:
        return list(map(int, union(token)))

class RedundantObjectToken(RedundantToken):
    '''
    Represents a segmentation object.
    '''
    def verify_token(self, token: Any) -> bool:
        return (
            isinstance(token, list) and
            all(
                isinstance(x, list) and
                all(
                    isinstance(y, tuple) and
                    isinstance(y[0], float) and
                    isinstance(y[1], float)
                    for y in x
                )
                for x in token
            )
        )
    def transform(self, token: list) -> Any:
        if len(token) > 0 and isinstance(token[0], list):
            return token
        return [token]

class UniqueIDToken(IDToken, UniqueToken):
    '''
    Represents a unique ID.
    '''

class DataType:
    '''
    All possible data types. Container class for Token objects with specific purposes.

     - `desc` (`str`): the purpose of the DataType. This should be unique for every new object.
     - `token_type` (`Token`): the token type of the DataType.
    '''

    def __init__(self, desc: str, token_type: Token) -> None:
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
        
        - `value` (`str`): the value to check if it is compatible with the DataType.
        '''
        return self.token_type.verify_token(value)

class DataTypes:
    '''
    Presets for DataType. These represent valid tokens, and DataType should not be initialized
    directly but rather through these presets.
    '''

    # main types
    IMAGE_SET_NAME = DataType('IMAGE_SET_NAME', RedundantToken())
    IMAGE_SET_ID = DataType('IMAGE_SET_ID', RedundantIDToken())
    ABSOLUTE_FILE = DataType('ABSOLUTE_FILE', FilenameToken())
    ABSOLUTE_FILE_SEG = DataType('ABSOLUTE_FILE_SEG', FilenameToken())
    IMAGE_NAME = DataType('IMAGE_NAME', UniqueToken())
    IMAGE_ID = DataType('IMAGE_ID', UniqueIDToken())
    GENERIC = DataType('GENERIC', WildcardToken())

    # classification
    CLASS_NAME = DataType('CLASS_NAME', Token())
    CLASS_ID = DataType('CLASS_ID', IDToken())

    # detection
    BBOX_CLASS_NAME = DataType('BBOX_CLASS_NAME', RedundantToken())
    BBOX_CLASS_ID = DataType('BBOX_CLASS_ID', RedundantIDToken())
    XMIN = DataType('XMIN', RedundantQuantityToken())
    YMIN = DataType('YMIN', RedundantQuantityToken())
    XMAX = DataType('XMAX', RedundantQuantityToken())
    YMAX = DataType('YMAX', RedundantQuantityToken())
    XMID = DataType('XMID', RedundantQuantityToken())
    YMID = DataType('YMID', RedundantQuantityToken())
    X1 = DataType('X1', RedundantQuantityToken())
    Y1 = DataType('Y1', RedundantQuantityToken())
    X2 = DataType('X2', RedundantQuantityToken())
    Y2 = DataType('Y2', RedundantQuantityToken())
    WIDTH = DataType('WIDTH', RedundantQuantityToken())
    HEIGHT = DataType('HEIGHT', RedundantQuantityToken())

    # segmentation
    SEG_CLASS_NAME = DataType('SEG_CLASS_NAME', RedundantToken())
    SEG_CLASS_ID = DataType('SEG_CLASS_ID', RedundantIDToken())
    X = DataType('X', RedundantQuantityToken())
    Y = DataType('Y', RedundantQuantityToken())
    POLYGON = DataType('POLYGON', RedundantObjectToken())

class DataItem:
    '''
    Base class for representing a data item. Contains a DataType and a value associated with it.

     - `delimiter` (`DataType`): the type of the DataItem.
     - `value` (`Any`): the value associated with the DataType, must be compatible.
    '''
    def __init__(self, delimiter: DataType, value: Any) -> None:
        value = delimiter.token_type.transform(value)
        if not delimiter.token_type.verify_token(value):
            Warnings.error('data_invalid', value=value, delimiter=delimiter)
        self.delimiter: DataType = delimiter
        self.value = value

    def __repr__(self) -> str:
        return f'{self.delimiter}: {self.value}'

    def __eq__(self, other: Self) -> bool:
        if self.__class__ != other.__class__:
            return False
        return self.delimiter == other.delimiter and self.value == other.value

    def add(self, item: Self) -> None:
        '''
        Add an item to current data if it is redundant.
        
         - `item` (`Self`): an item to add to itself.
        
        Throws `ValueError` if either `self` or `item` are not redundant.
        '''
        if (not isinstance(self.delimiter.token_type, RedundantToken) or
            not isinstance(item.delimiter.token_type, RedundantToken)):
            Warnings.error('nonredundant_add')
        self.value = self.value + item.value

    @classmethod
    def copy(cls, first: Self) -> Self:
        '''
        Shallow copy self's data into new instance.
        
         - `first` (`Self`): item to copy.
        '''
        return cls(first.delimiter, copy(first.value))

class DataEntry:
    '''
    Contains all items required for an entry in the dataset, which contains DataItem objects.
    
     - `items` (`list[DataItem] | DataItem`): a (list of) data items which are to be batched together
    '''

    _valid_sets = [
        {'IMAGE_SET_ID', 'IMAGE_SET_NAME'},
        {'XMIN', 'XMAX', 'YMIN', 'YMAX', 'XMID', 'YMID', 'X1', 'X2', 'Y1', 'Y2', 'WIDTH', 'HEIGHT',
         'BBOX_CLASS_ID', 'BBOX_CLASS_NAME'},
        {'POLYGON', 'SEG_CLASS_ID', 'SEG_CLASS_NAME'},
        {'X', 'Y'}
    ]

    def __init__(self, items: Union[list[DataItem], DataItem]) -> None:
        items: list[DataItem] = union(items)
        self.unique: bool = any(isinstance(item.delimiter.token_type, UniqueToken)
                                for item in items if not isinstance(item, list))
        self.data: dict[str, DataItem] = {(item.delimiter.desc if not isinstance(item, list)
                                           else item[0].delimiter.desc): item for item in items}

    @classmethod
    def merge(cls, first: Self, second: Self) -> Self:
        '''
        Merge two data entries together, storing it in a new instance. For inplace operations see
        `merge_inplace`.
        
         - `first` (`DataEntry`): the first data entry to merge.
         - `second` (`DataEntry`): the second data entry to merge.
         - `overlap` (`bool`): whether to allow nonunique item overlapping. Default: True.
        '''
        merged = cls(list(first.data.values()))
        redundant_overlap = set()
        for desc, item in second.data.items():
            if isinstance(item.delimiter.token_type, WildcardToken):
                continue
            if isinstance(item.delimiter.token_type, RedundantToken):
                if desc in merged.data and merged.data[desc] != second.data[desc]:
                    redundant_overlap.add(desc)
                continue
            if desc in merged.data and merged.data[desc] != second.data[desc]:
                Warnings.error('merge_conflict', first=first, second=second)
        allocated = False
        for group in DataEntry._valid_sets:
            if redundant_overlap.issubset(group):
                redundant_overlap = group
                allocated = True
                break
        if not allocated:
            Warnings.error(
                'merge_redundant_conflict',
                overlap=redundant_overlap,
                first=first,
                second=second
            )
        for desc in redundant_overlap:
            if desc in merged.data and desc in second.data:
                merged.data[desc].add(second.data[desc])
        for desc, item in second.data.items():
            if desc not in merged.data:
                merged.data[desc] = item
                continue
        return merged

    def merge_inplace(self, other: Self) -> None:
        '''
        Merge two data entries together, storing it in this instance.
        
         - `other` (`DataEntry`): the other data entry to merge into this instance.
         - `overlap` (`bool`): whether to allow nonunique item overlapping. Default: True.
        '''
        redundant_overlap = set()
        for desc, item in other.data.items():
            if isinstance(item.delimiter.token_type, WildcardToken):
                continue
            if isinstance(item.delimiter.token_type, RedundantToken):
                if desc in self.data and self.data[desc] != other.data[desc]:
                    redundant_overlap.add(desc)
                continue
            if desc in self.data and self.data[desc] != other.data[desc]:
                Warnings.error('merge_conflict', first=self, second=other)
        allocated = False
        for group in DataEntry._valid_sets:
            if redundant_overlap.issubset(group):
                redundant_overlap = group
                allocated = True
                break
        if not allocated:
            Warnings.error(
                'merge_redundant_conflict',
                overlap=redundant_overlap,
                first=self,
                second=other
            )
        for desc in redundant_overlap:
            if desc in self.data and desc in other.data:
                self.data[desc].add(other.data[desc])
        for desc, item in other.data.items():
            if desc not in self.data:
                self.data[desc] = item
                continue

    def apply_tokens(self, items: Union[list[DataItem], DataItem]) -> None:
        '''
        Apply new tokens to the item.
        
         - `items` (`list[DataItem] | DataItem`): additional items to associate with this data
            entry.
        '''
        items: list[DataItem] = [DataItem.copy(item) for item in union(items)]
        # execute checks first
        for item in items:
            if isinstance(item.delimiter.token_type, RedundantToken):
                continue
            if isinstance(item.delimiter.token_type, UniqueToken):
                if self.data[item.delimiter.desc] != item:
                    Warnings.error(
                        'merge_unique_conflict',
                        parent=self.data[item.delimiter.desc],
                        data=item
                    )
        # merge
        for item in items:
            if item.delimiter.desc not in self.data:
                if not isinstance(item.delimiter.token_type, RedundantToken):
                    self.data[item.delimiter.desc] = item
                    continue
                for group in DataEntry._valid_sets:
                    if item.delimiter.desc in group:
                        break
                # redundant token must fall into one of these groups so no error checking
                # if none of the groups already exist then default to 1x application otherwise
                # must match length with other items in the group
                n = 1
                for desc in group:
                    if desc in self.data:
                        n = len(self.data[desc].value)
                        break

                assert len(item.value) == 1, \
                    ('Assertion failed - (len(item.value) == 1) Please report this error to the '
                     'CVData developers.')

                self.data[item.delimiter.desc] = DataItem(
                    item.delimiter,
                    item.value * n
                )
                continue
            elif isinstance(item.delimiter.token_type, RedundantToken):
                self.data[item.delimiter.desc].add(item)

    def __repr__(self) -> str:
        return ' '.join(['DataEntry:']+[str(item) for item in self.data.values()])

class Alias:
    '''
    Class used when a DataType placeholder could be interpreted multiple ways. For example, if
    IMAGE_NAME also contains CLASS_NAME and IMAGE_ID, we can extract all 3 tokens out using
    PatternAlias. Counts for a single wildcard token (`{}`).
    
     - `generics` (`list[Generic | DataType]`): the list of Generic type objects which can be used
        for alias parsing.
    '''
    def __init__(self, generics: list[Union['Generic', DataType]]) -> None:
        if len(generics) == 0:
            Warnings.error('generics_missing')
        self.generics = [generic if isinstance(generic, Generic)
                         else Generic('{}', DataType) for generic in generics]
        self.patterns: list[str] = [generic.pattern for generic in generics]
        self.aliases: list[tuple[DataType, ...]] = [generic.data for generic in generics]
        self.desc = ''.join([token.desc for alias in self.aliases for token in alias])

    def match(self, entry: str) -> tuple[bool, list[DataItem]]:
        '''
        Return a list of DataItems including all of the possible alias items.
        
        - `entry` (`str`): the entry to be matched with for the alias.
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
            except ValueError:
                return False, []
        return True, result

    def __repr__(self) -> str:
        return str(dict(zip(self.patterns, self.aliases)))

class Static:
    '''
    Represents an object with a static name. Can contain data.
    
     - `name` (`str`): the value associated with the Static.
     - `data` (`DataItem | list[DataItem]`): the data item(s) associated with the name.
    '''
    def __init__(
        self,
        name: str,
        data: Optional[Union[list[DataItem], DataItem]] = None
    ) -> None:
        self.name: str = name
        if data is None:
            data = []
        self.data: list[DataItem] = union(data)

    def match(self, entry: str) -> tuple[bool, list[DataItem]]:
        '''
        Checks if the entry string matches this static item.
        
         - `entry` (`str`): the entry string to check for equality
        '''
        matched: bool = entry == self.name
        data: list[DataItem] = self.data if matched else []
        return matched, data

    def __repr__(self) -> str:
        return f'*-{self.name} ({", ".join([str(item) for item in self.data])})-*'

class Generic:
    '''
    Represents an object with a generic name and datatypes fitted into wildcards.
    
     - `pattern` (`str | DataType | Alias`): the pattern with which to match to, containing
        wildcards of the `{}` format. It is assumed that the generic should be matched to the entire
        string. Regex expressions compatible with the `re` module are allowed except capture groups
        such as `(.+)`, which will throw an error.
     - `data` (`DataType | Alias`): tokens that correspond to data types which each `{}` matches to.
     - `ignore` (`list[str] | str`): values that begin with any item in `ignore` are not matched.
    '''
    def __init__(
        self,
        pattern: Union[str, DataType, Alias],
        *data: Union[DataType, Alias],
        ignore: Optional[Union[list[str], str]] = None
    ) -> None:
        if isinstance(pattern, (DataType, Alias)):
            data = tuple([pattern])
            pattern = '{}'
        if len(data) != pattern.count('{}'):
            Warnings.error(
                'row_mismatch',
                name1='wildcard groups',
                name2='DataType tokens',
                len1=pattern.count('{}'),
                len2=len(data)
            )
        if '(.+)' in pattern:
            Warnings.error('illegal_capturing_group')
        self.pattern: str = '^' + pattern.replace('{}', '(.+)') + '+$'
        self.data: tuple[Union[DataType, Alias], ...] = data
        if ignore is None:
            ignore = []
        self.ignore: list[str] = ['^' + ignore_pattern.replace('{}', '(.+)') + '+$'
                                  for ignore_pattern in union(ignore)]

    def match(self, entry: str) -> tuple[bool, list[DataItem]]:
        '''
        Return a list of the tokens' string values provided an entry string which follows the 
        pattern.
        
         - `entry` (`str`): the string to match to the pattern, assuming it does match
        '''
        for ignore_pattern in self.ignore:
            if re.findall(ignore_pattern, entry):
                return False, []
        matches: list[str] = re.findall(self.pattern, entry)
        result: list[DataItem] = []

        if not matches:
            return False, []

        try:
            if isinstance(matches[0], tuple):
                matches = matches[0]
            for data_type, match in zip(self.data, matches):
                if not isinstance(data_type, Alias):
                    result.append(DataItem(data_type, match))
                    continue
                success, matched = data_type.match(match)
                if not success:
                    return False, []
                result += matched
        except ValueError:
            return False, []
        return True, result

    def __repr__(self) -> str:
        return f'{self.pattern} | {self.data}'

class Folder(Generic):
    '''
    Generic for directories only.
    '''

class File(Generic):
    '''
    Generic for files only.
    
     - `pattern` (`str | DataType | Alias`): the pattern with which to match to, containing
        wildcards of the `{}` format. It is assumed that the generic should be matched to the entire
        string. Regex expressions compatible with the `re` module are allowed except capture groups
        such as `(.+)`, which will throw an error.
     - `data` (`DataType | Alias`): tokens that correspond to data types which each `{}` matches to.
     - `ignore` (`list[str] | str`): values that begin with any item in `ignore` are not matched.
        For filenames, this should not include the file extension.
     - `extensions` (`list[str] | str`): valid extensions to match to. This will be whatever is
        after the `.`, i.e. `txt`. Files without extensions are not allowed, but can be used as a
        Generic.
     - `disable_warnings` (`bool`): disables the warnings that incur when `pattern` includes `.` in
        it. This may be useful when the filenames do indeed include `.` without it being the ext.
    '''
    def __init__(
        self,
        pattern: Union[str, DataType, Alias],
        *data: Union[DataType, Alias],
        ignore: Optional[Union[list[str], str]] = None,
        extensions: Union[list[str], str] = '',
        disable_warnings: bool = False
    ) -> None:
        extensions = list(map(lambda s: s.lower(), union(extensions)))
        if isinstance(pattern, (DataType, Alias)):
            data = tuple([pattern])
            pattern = '{}'
        result = re.findall('(.+)\.(.+)', pattern)
        if not disable_warnings and result:
            Warnings.warn('file_ext')
        self.extensions = extensions
        super().__init__(pattern, *data, ignore=ignore)

    def match(self, entry: str) -> tuple[bool, list[DataItem]]:
        '''
        Return a list of the tokens' string values provided an entry string which follows the 
        pattern.
        
         - `entry` (`str`): the string to match to the pattern, assuming it does match
        '''
        result = re.findall('(.+)\.(.+)', entry)
        if not result:
            return False, []
        if self.extensions and (result[0][1].lower() not in self.extensions):
            return False, []
        return super().match(result[0][0])

class ImageFile(File):
    '''
    Generic with special matching for images. Uses a sublist of valid image extensions which are
    supported by the `PIL.Image` library.
    '''
    _image_extensions = ['jpg', 'jpeg', 'png', 'tiff', 'jpe', 'jfif', 'j2c', 'j2k', 'jp2', 'jpc',
                         'jpf', 'jpx', 'apng', 'tif', 'webp']

    def __init__(
        self,
        pattern: str,
        *data: Union[DataType, Alias],
        ignore: Optional[Union[list[str], str]] = None,
        extensions: Optional[Union[list[str], str]] = None
    ) -> None:
        if extensions is None:
            extensions = ImageFile._image_extensions
        super().__init__(pattern, *data, ignore=ignore, extensions=extensions)

class Image:
    '''
    Generic image, used as a value in the dict to represent Image objects.
    '''
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "Image"

class SegmentationImage:
    '''
    Segmentation mask image. The image should contain pixels which refer to a mask for segmentation.
    '''
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "Segmentation Image"
