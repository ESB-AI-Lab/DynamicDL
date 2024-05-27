'''
.. module:: DataItem

'''

from copy import copy
from typing import Any
from typing_extensions import Self

from .._warnings import Warnings
from .tokens import RedundantToken
from .datatype import DataType

class DataItem:
    '''
    The `DataItem` class represents a value associated with a particular `DataType`. DataItem
    objects are regularly handled and created by internal processes, but can be used in
    instantiating `Static` variables with certain values.
    
    Example:
    `my_static = Static('my_image_set_name', DataItem(DataTypes.IMAGE_SET_NAME), 'my_set')`
    The above example creates a static which contains the value `my_set` as an image set name for
    its hierarchical children to inherit.

    :param delimiter: The type of the DataItem.
    :type delimiter: DataType
    :param value: The value associated with the DataType, must be compatible.
    :type value: Any
    '''
    def __init__(self, delimiter: DataType, value: Any) -> None:
        value = delimiter.token_type.transform(value)
        if not delimiter.token_type.verify_token(value):
            Warnings.error('data_invalid', value=value, delimiter=delimiter)
        self.delimiter: DataType = delimiter
        self.value = value

    def __repr__(self) -> str:
        return f'{self.delimiter.desc}: {self.value}'

    def __eq__(self, other: Self) -> bool:
        if self.__class__ != other.__class__:
            return False
        return self.delimiter == other.delimiter and self.value == other.value

    def add(self, item: Self) -> None:
        '''
        Add an item to current data if it is redundant. Used by internal merging processes.
        
        :param item: An item to add to itself.
        :type item: DataItem
        :raises ValueError: Either `self` or `item` are not redundant.
        '''
        if (not isinstance(self.delimiter.token_type, RedundantToken) or
            not isinstance(item.delimiter.token_type, RedundantToken)):
            Warnings.error('nonredundant_add')
        self.value = self.value + item.value

    @classmethod
    def copy(cls, first: Self) -> Self:
        '''
        Shallow copy self's data into new instance. Used by internal merging processes.
        
        :param first: The item to copy.
        :type first: DataItem
        :return: A shallow copy of the data item.
        :rtype: DataItem
        '''
        return cls(first.delimiter, copy(first.value))