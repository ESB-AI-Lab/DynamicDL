'''
Generic type objects.
'''
from __future__ import annotations

import re
from typing import Union, Optional, TYPE_CHECKING

from .._utils import union
from .._warnings import Warnings
from ..data.datatype import DataType
from ..data.dataitem import DataItem

if TYPE_CHECKING:
    from .alias import Alias

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
    
    :param pattern: The pattern with which to match to, containing wildcards  of the `{}` format. It
        is assumed that the generic should be matched to the entire string. Regex expressions
        compatible with the `re` module are allowed except capture groups such as `(.+)`, which will
        throw an error. If `DataType` or `Alias` is specified, data is overriden and has no effect.
    :type pattern: str | DataType | Alias
    :param data: Tokens that correspond to data types which each `{}` matches to.
    :type data: DataType | Alias
    :param ignore: Values that match any item in `ignore` are not matched. Currently only supports
        str, in future versions will support Generic types.
    :type ignore: list[str] | str
    :raises LengthMismatchError: The length of the `{}` wildcards must match the number of DataType
        or Alias values provided in `data`.
    :raises ValueError: (.+) and (.*) regex groups cannot be present in the pattern; use `{}` with
        an associated DataType instead.
    '''
    def __init__(
        self,
        pattern: Union[str, DataType, Alias],
        *data: Union[DataType, Alias],
        ignore: Optional[Union[list[str], str]] = None
    ) -> None:
        if not isinstance(pattern, str):
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
        if '(.+)' in pattern or '(.*)' in pattern:
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
        
        :param entry: The entry string to be matched to the generic pattern.
        :type entry: str
        :return: A boolean indicating success of the matching, and a list of the DataItems passed.
        :rtype: tuple[bool, list[DataItem]]
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
                if isinstance(data_type, DataType):
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
        return f'G[{self.pattern[1:-2].replace("(.+)", "{}")} | {self.data}]'

class Folder(Generic):
    '''
    A subclass of `Generic` which extends Generic pattern matching but for valid directories in the
    filesystem only. During parsing, `Folder` must be parsed as keys in the filestructure format.
    All behaviors are otherwise exactly alike.
    
    :param pattern: The pattern with which to match to, containing wildcards  of the `{}` format. It
        is assumed that the generic should be matched to the entire string. Regex expressions
        compatible with the `re` module are allowed except capture groups such as `(.+)`, which will
        throw an error. If `DataType` or `Alias` is specified, data is overriden and has no effect.
    :type pattern: str | DataType | Alias
    :param data: Tokens that correspond to data types which each `{}` matches to.
    :type data: DataType | Alias
    :param ignore: Values that match any item in `ignore` are not matched. Currently only supports
        str, in future versions will support Generic types.
    :type ignore: list[str] | str
    '''

class File(Generic):
    '''
    A subclass of `Generic` which extends Generic pattern matching but for valid files in the
    filesystem only. During parsing, `File` must be parsed as keys in the filestructure format.
    All behaviors are otherwise exactly alike. Also takes a list of valid extensions. In future
    versions, filetypes will be inferred from the corresponding value in the filestructure format.
    
    :param pattern: The pattern with which to match to, containing wildcards  of the `{}` format. It
        is assumed that the generic should be matched to the entire string. Regex expressions
        compatible with the `re` module are allowed except capture groups such as `(.+)`, which will
        throw an error. If `DataType` or `Alias` is specified, data is overriden and has no effect.
    :type pattern: str | DataType | Alias
    :param data: Tokens that correspond to data types which each `{}` matches to.
    :type data: DataType | Alias
    :param ignore: Values that match any item in `ignore` are not matched. Currently only supports
        str, in future versions will support Generic types.
    :type ignore: list[str] | str
    :param extensions: Valid extensions to match to. This will be whatever is after the `.`, i.e.
        `txt`. Files without extensions are not allowed, but can be instead parsed as a Generic.
    :type extensions: list[str] | str
    :param disable_warnings: Disables the warnings that incur when `pattern` includes `.` in it.
        This may be useful when the filenames do indeed include `.` without it being the ext.
    :type disable_warnings: bool
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
        if not isinstance(pattern, str):
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
        
        :param entry: The entry string to be matched to the generic pattern.
        :type entry: str
        :return: A boolean indicating success of the matching, and a list of the DataItems passed.
        :rtype: tuple[bool, list[DataItem]]
        '''
        result = re.findall('(.+)\.(.+)', entry)
        if not result:
            return False, []
        if self.extensions and (result[0][1].lower() not in self.extensions):
            return False, []
        return super().match(result[0][0])

class ImageFile(File):
    '''
    A subclass of `File` which extends Generic pattern matching but for valid images in the
    filesystem only. During parsing, `ImageFile` must be parsed as keys in the filestructure format.
    All behaviors are otherwise exactly alike. Default image extensions are provided but can also be
    specified to restrict to a certain subset. In the future, this class may be deprecated to
    support automatic type inference.
    
    :param pattern: The pattern with which to match to, containing wildcards  of the `{}` format. It
        is assumed that the generic should be matched to the entire string. Regex expressions
        compatible with the `re` module are allowed except capture groups such as `(.+)`, which will
        throw an error. If `DataType` or `Alias` is specified, data is overriden and has no effect.
    :type pattern: str | DataType | Alias
    :param data: Tokens that correspond to data types which each `{}` matches to.
    :type data: DataType | Alias
    :param ignore: Values that match any item in `ignore` are not matched. Currently only supports
        str, in future versions will support Generic types.
    :type ignore: list[str] | str
    :param extensions: Valid extensions to match to. This will be whatever is after the `.`, i.e.
        `txt`. Files without extensions are not allowed, but can be instead parsed as a Generic.
    :type extensions: list[str] | str
    :param disable_warnings: Disables the warnings that incur when `pattern` includes `.` in it.
        This may be useful when the filenames do indeed include `.` without it being the ext.
    :type disable_warnings: bool
    '''
    _image_extensions = ['jpg', 'jpeg', 'png', 'tiff', 'jpe', 'jfif', 'j2c', 'j2k', 'jp2', 'jpc',
                         'jpf', 'jpx', 'apng', 'tif', 'webp']

    def __init__(
        self,
        pattern: str,
        *data: Union[DataType, Alias],
        ignore: Optional[Union[list[str], str]] = None,
        extensions: Optional[Union[list[str], str]] = None,
        disable_warnings: bool = False
    ) -> None:
        if extensions is None:
            extensions = ImageFile._image_extensions
        super().__init__(
            pattern,
            *data,
            ignore=ignore,
            extensions=extensions,
            disable_warnings=disable_warnings
        )
