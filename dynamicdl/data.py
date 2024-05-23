'''
Module with fundamental data classes for dataset parsing.
'''

from copy import copy
from typing import Any, Union, Iterable
from typing_extensions import Self

from ._utils import union, Warnings
from .tokens import Token, UniqueToken, UniqueIDToken, RedundantIDToken, RedundantObjectToken, \
                    RedundantQuantityToken, RedundantToken, WildcardIntToken, \
                    WildcardQuantityToken, WildcardToken, WildcardWordToken, FilenameToken, IDToken

__all__ = [
    'DataType',
    'DataTypes',
    'DataItem',
    'DataEntry',
]

class DataType:
    '''
    `DataType` is a container class for storing relevant dataset items. As of 0.1.1-alpha, it is not
    yet supported for users to create their own `DataType` objects. Instead, currently usage is
    through the `DataTypes` module. This will change in future versions. Static objects and further
    documentation are provided in the `DataTypes` class.

    Constructor parameters:
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
    The `DataTypes` class contains static presets for `DataType` types. Below is a description of
    all presets currently available:
    
    General DataTypes
    +++++++++++++++++
    
    - `IMAGE_SET_NAME`: Represents the name of an image set. This includes any valid strings, but
      is not meant to store the ID of the image set; see `IMAGE_SET_ID`. Image sets are used to
      allocate specific entries to a group which can be split when dataloading. Most commonly
      image set names will be `train`, `val`, or `test`.
    - `IMAGE_SET_ID`: Represents the ID of an image set. This includes any valid integers. The named
      complement of this DataType is `IMAGE_SET_NAME`. See above for details.
    - `ABSOLUTE_FILE`: Represents the **absolute** filepath of an entry image only. This DataType is
      automatically generated in `Image` and `File` type objects when parsing, but can also be
      used to parse data. All valid values under `ABSOLUTE_FILE` must be a valid filepath on the
      user's filesystem. `RELATIVE_FILE` is currently not supported, but may be in future versions.
    - `ABSOLUTE_FILE_SEG`: Represents the **absolute** filepath of an entry segmentation mask only.
      This DataType is also automatically generated in `Image` and `File` type objects when
      parsing, but can also be used to parse data. All valid values under `ABSOLUTE_FILE` must be
      a valid filepath on the user's filesystem. `RELATIVE_FILE_SEG` is currently not supported,
      but may be in future versions.
    - `IMAGE_NAME`: Represents an identifier token for image entries via a string description.
      As of 0.1.1-alpha all `IMAGE_NAME` entries must be unique as it serves as a sole identifier
      for image entries. Accepts parsed strings. Its ID complement can be found under `IMAGE_ID`.
    - `IMAGE_ID`: The ID (parsed to int) complement for `IMAGE_NAME`. Behaves just like its
      complement.
    - `GENERIC`: A generic token with no significance that can be used as a wildcard token for
      parsing. Can represent anything, and any type.
    - `GENERIC_INT`: Same as `GENERIC`, except accepts only integer types.
    - `GENERIC_QUANTITY`: Same as `GENERIC`, except accepts only numeric types (i.e. float and int)
    - `GENERIC_WORD`: Same as `GENERIC`, except accepts only one word, i.e. no spaces allowed.
    
    Classification DataTypes
    ++++++++++++++++++++++++
    
    - `CLASS_NAME`: Represents the classification class name of an image entry. There can only be
      one class per image entry, and accepts parsed strings. Its ID complement can be found under
      `CLASS_ID`.
    - `CLASS_ID`: The ID (parsed to int) complement for `CLASS_NAME`. Behaves just like its
      complement.
    
    Detection DataTypes
    +++++++++++++++++++
    
    - `BBOX_CLASS_NAME`: Represents the detection class name of an image entry. There can be
      multiple classes per image entry, and accepts parsed strings. Its ID complement can be found
      under `BBOX_CLASS_ID`. Each detection class must have a one-to-one correspondence to a valid
      bounding box when in the same hierarchy. When in different hierarchies it, just like other
      redundant types, will expand naturally to fit the existing length.
    - `BBOX_CLASS_ID`: The ID (parsed to int) complement for `BBOX_CLASS_NAME`. Behaves just like
      its complement.
    - `XMIN`: The minimum x-coordinate in the bounding box. Must be accompanied with `YMIN` or else
      has no effect, and must be accompanied either with `XMAX` or `WIDTH` and their y-counterparts.
    - `YMIN`: The minimum y-coordinate in the bounding box. Must be accompanied with `XMIN` or else
      has no effect, and must be accompanied either with `YMAX` or `HEIGHT` and their
      x-counterparts.
    - `XMAX`: The maximum x-coordinate in the bounding box. Must be accompanied with `YMAX` or else
      has no effect, and must be accompanied either with `XMIN` or `WIDTH` and their y-counterparts.
    - `YMAX`: The maximum y-coordinate in the bounding box. Must be accompanied with `XMAX` or else
      has no effect, and must be accompanied either with `YMIN` or `HEIGHT` and their
      x-counterparts.
    - `XMID`: The midpoint x-coordinate in the bounding box. Used to denote the horizontal center of
      the bounding box. Must be accompanied with `YMID` to define a central point.
    - `YMID`: The midpoint y-coordinate in the bounding box. Used to denote the vertical center of
      the bounding box. Must be accompanied with XMID to define a central point.
    - `X1`: A bounding box x-coordinate. Can be in any order as long as it forms a valid bounding
      box with `Y1`, `X2`, and `Y2`.
    - `Y1`: A bounding box y-coordinate. Can be in any order as long as it forms a valid bounding
      box with `X1`, `X2`, and `Y2`.
    - `X2`: A bounding box x-coordinate. Can be in any order as long as it forms a valid bounding
      box with `Y1`, `X1`, and `Y2`.
    - `Y2`: A bounding box y-coordinate. Can be in any order as long as it forms a valid bounding
      box with `X1`, `X2`, and `Y1`.
    - `WIDTH`: The width of the bounding box. Must be accompanied with `HEIGHT` or else has no
      effect. Can be used as an alternative to defining `XMAX` and `XMIN`.
    - `HEIGHT`: The height of the bounding box. Must be accompanied with `WIDTH` or else has no
      effect. Can be used as an alternative to defining `YMAX` and `YMIN`.
    
    Segmentation DataTypes
    ++++++++++++++++++++++

    - `SEG_CLASS_NAME`: Represents the segmentation class name of an image entry. There can be
      multiple classes per image entry, and accepts parsed strings. Its ID complement can be found
      under `SEG_CLASS_ID`. Each detection class must have a one-to-one correspondence to a valid
      bounding box when in the same hierarchy. When in different hierarchies it, just like other
      redundant types, will expand naturally to fit the existing length.
    - `SEG_CLASS_ID`: The ID (parsed to int) complement for `SEG_CLASS_NAME`. Behaves just like
      its complement.
    - `X`: A segmentation polygon x-coordinate. Used to define the vertices of a polygon for
      segmentation tasks. Each `X` coordinate must be paired with a corresponding `Y` coordinate
      to form a valid vertex.
    - `Y`: A segmentation polygon y-coordinate. Used to define the vertices of a polygon for
      segmentation tasks. Each `Y` coordinate must be paired with a corresponding `X` coordinate
      to form a valid vertex.
    - `POLYGON`: Should not be instantiated by the user as there is no way to parse it. However, it
      is automatically created upon every `SegmentationObject` wrapper of `X` and `Y` objects.
      This DataType is used internally for parsing.
    
    '''

    # main types
    IMAGE_SET_NAME = DataType('IMAGE_SET_NAME', RedundantToken())
    IMAGE_SET_ID = DataType('IMAGE_SET_ID', RedundantIDToken())
    ABSOLUTE_FILE = DataType('ABSOLUTE_FILE', FilenameToken())
    ABSOLUTE_FILE_SEG = DataType('ABSOLUTE_FILE_SEG', FilenameToken())
    IMAGE_NAME = DataType('IMAGE_NAME', UniqueToken())
    IMAGE_ID = DataType('IMAGE_ID', UniqueIDToken())
    GENERIC = DataType('GENERIC', WildcardToken())
    GENERIC_INT = DataType('GENERIC_INT', WildcardIntToken())
    GENERIC_QUANTITY = DataType('GENERIC_QUANTITY', WildcardQuantityToken())
    GENERIC_WORD = DataType('GENERIC_WORD', WildcardWordToken())

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
    The `DataItem` class represents a value associated with a particular `DataType`. DataItem
    objects are regularly handled and created by internal processes, but can be used in
    instantiating `Static` variables with certain values.
    
    Example:
    `my_static = Static('my_image_set_name', DataItem(DataTypes.IMAGE_SET_NAME), 'my_set')`
    The above example creates a static which contains the value `my_set` as an image set name for
    its hierarchical children to inherit.

    Constructor parameters:
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
        Add an item to current data if it is redundant. Used by internal merging processes.
        
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
        Shallow copy self's data into new instance. Used by internal merging processes.
        
        - `first` (`Self`): item to copy.
        '''
        return cls(first.delimiter, copy(first.value))

class DataEntry:
    '''
    Contains all items required for an entry in the dataset, a collection of DataItem objects. Most
    use is handled by internal merging processes, and is not to be instantiated by users.
    
    Constructor parameters:
    
    - `items` (`list[DataItem] | DataItem`): a (list of) data items which are to be batched
        together

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
        self.data: dict[str, DataItem] = {item.delimiter.desc: item for item in items}
        self._update_unique()

    def _update_unique(self) -> bool:
        self.unique = any(isinstance(item.delimiter.token_type, UniqueToken)
            for item in self.data.values())

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
        merged._update_unique()
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
        self._update_unique()

    def apply_tokens(self, items: Union[list[DataItem], DataItem]) -> None:
        '''
        Apply new tokens to the item.
        
        - `items` (`list[DataItem] | DataItem`): additional items to associate with this data
            entry.
        '''
        if not isinstance(items, Iterable):
            items = [items]
        items: list[DataItem] = [DataItem.copy(item) for item in items]
        # execute checks first
        for item in items:
            if isinstance(item.delimiter.token_type, RedundantToken):
                continue
            if isinstance(item.delimiter.token_type, UniqueToken):
                if item.delimiter.desc in self.data and self.data[item.delimiter.desc] != item:
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
                matched = False
                for desc in group:
                    if desc in self.data:
                        n = len(self.data[desc].value)
                        matched = True
                        break
                assert not matched or len(item.value) == 1 or n == len(item.value), \
                    ('Assertion failed (report as a bug!) - (len(item.value) == 1);'
                     f'item: {item} | group: {group} | self: {self}')

                if not matched or n == len(item.value):
                    self.data[item.delimiter.desc] = DataItem(
                        item.delimiter,
                        item.value
                    )
                    continue
                self.data[item.delimiter.desc] = DataItem(
                    item.delimiter,
                    item.value * n
                )
            elif isinstance(item.delimiter.token_type, RedundantToken):
                self.data[item.delimiter.desc].add(item)
        self._update_unique()

    def __repr__(self) -> str:
        return ' '.join(['DataEntry:']+[str(item) for item in self.data.values()])
