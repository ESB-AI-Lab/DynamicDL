'''
Module with main data items for creating dataset formats.
'''

import re
from typing import Union, Optional

from ._utils import union, Warnings
from .data import DataItem, DataType

__all__ = [
    'Alias',
    'Static',
    'Generic',
    'Folder',
    'File',
    'ImageFile',
    'Image',
    'SegmentationImage'
]
class Alias:
    '''
    Class used when a placeholder in `Generic` could be interpreted multiple ways. For example, if
    `IMAGE_NAME` also contains `CLASS_NAME` and `IMAGE_ID`, we can extract all 3 tokens out using
    `Alias`. Counts for a single wildcard token (`{}`) when supplied in `Generic`. 
    
    Example:
    
    .. code-block:: python
    
        alias =  Alias([
            DataTypes.IMAGE_NAME,
            Generic('{}_{}', DataTypes.CLASS_NAME, DataTypes.IMAGE_ID)
        ])

    Now we can use `Generic(alias)` as a valid generic and it will obtain all contained DataTypes.
    
    Constructor parameters:
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
        Return a list of DataItems if matched successfully. Used for internal processing functions.
        
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
    Represents an object with a static name. Can contain data in the form of `DataItem` objects.
    
    Constructor parameters:
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
        Return status and DataItem objects (optional) if matched successfully. Used for internal
        processing functions.
        
        - `entry` (`str`): the entry to be matched with for the static.
        '''
        matched: bool = entry == self.name
        data: list[DataItem] = self.data if matched else []
        return matched, data

    def __repr__(self) -> str:
        return f'*-{self.name} ({", ".join([str(item) for item in self.data])})-*'

class Generic:
    '''
    The `Generic` class is a basic building block for representing wildcard-optional data. It can be
    used anywhere in the DynamicDL dataset format and provides the structure needed to interpret
    data items and tokens.
    
    Example:
    
    .. code-block:: python

        # example 1
        gen = Generic('{}_{}', DataTypes.IMAGE_SET_NAME, DataTypes.IMAGE_SET_ID)
        
        my_data_type = DataTypes.GENERIC
        # example 2
        Generic('{}', my_data_type)
        # example 3
        Generic(my_data_type)
        # example 4
        my_data_type
        
        # example 5
        Generic(
            '{}_{}',
            DataTypes.IMAGE_SET_NAME,
            DataTypes.IMAGE_SET_ID,
            ignore = [
                'invalid_line',
                '{}_invalidclasstype'
            ]
        )

    Above, we can see that example 1 allows items of `"*_*"` to be interpreted, where the first
    wildcard is interpreted as image set name, and the latter as image set id. The Generic class
    also accepts DataType, which is meant to encapsulate the full wildcard; in other words,
    example 2, 3, 4 are functionally the same.

    `Generic` also accepts a `ignore` kwarg parameter which is either a string or list of strings
    containing patterns where anything which matches will be ignored, accepting regex patterns and
    also using `{}` as a valid wildcard. This is illustrated in example 5.

    **Constructor parameters:**
    
    - `pattern` (`str | DataType | Alias`): the pattern with which to match to, containing wildcards
      of the `{}` format. It is assumed that the generic should be matched to the entire string.
      Regex expressions compatible with the `re` module are allowed except capture groups such as
      `(.+)`, which will throw an error.
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

class Namespace:
    '''
    Generic which contains a set of valid names.
    '''
    def __init__(self, *names: Union[str, Static]):
        self.names = names

    def match(self, entry: str):
        for name in self.names:
            if isinstance(name, str):
                if entry == name:
                    return True, []
                continue
            if name.name == entry:
                return True, name.data
        return False, []

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
