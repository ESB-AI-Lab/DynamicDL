
from typing import Optional, Union

from .._utils import union
from ..data.datatype import DataType
from ..data.dataitem import DataItem

class Static:
    '''
    Represents an object with a static name. Can contain data in the form of `DataItem` objects.
    
    Example:
    
    .. code-block:: python
    
        # instantiate a static with specific data item
        Static('specific_name', DataItem(DataTypes.SOME_TYPE, 'some_other_specific_data'))
        
        # instantiate a static with the name inferred as a data type
        Static('specific_name_as_data', DataTypes.SOME_TYPE)

    :param name: The value associated with the Static.
    :type name: str
    :param data: The data item(s) associated with the name. Alternatively, can provide a single
        DataType which is inferred to be associated with `name`.
    :type data: DataItem | list[DataItem] | DataType
    '''
    def __init__(
        self,
        name: str,
        data: Optional[Union[list[DataItem], DataItem, DataType]] = None
    ) -> None:
        self.name: str = name
        if isinstance(data, DataType):
            data = DataItem(data, name)
        if data is None:
            data = []
        self.data: list[DataItem] = union(data)

    def match(self, entry: str) -> tuple[bool, list[DataItem]]:
        '''
        Return status and DataItem objects (optional) if matched successfully. Used for internal
        processing functions. The return values are to be consistent with internal processing by
        Generics.
        
        :param entry: The entry string to be matched to the static pattern.
        :type entry: str
        :return: A boolean indicating success of the matching, and a list of the DataItems passed.
        :rtype: tuple[bool, list[DataItem]]
        '''
        matched: bool = entry == self.name
        data: list[DataItem] = self.data if matched else []
        return matched, data

    def __repr__(self) -> str:
        return f'[{self.name} ({", ".join([str(item) for item in self.data])})]'
    