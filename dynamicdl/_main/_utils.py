from typing import Union

from ..data.tokens import UniqueToken
from ..data.datatype import DataType
from ..data.dataitem import DataItem
from ..data.dataentry import DataEntry
from ..parsing.static import Static

def unique(item: Union[DataType, DataItem, DataEntry]) -> Union[str, list[str]]:
    '''
    Validate whether an item contains a unique identifier.
    '''
    if isinstance(item, DataEntry):
        return [data_item.delimiter.desc for data_item in item.data.values() if unique(data_item)]
    if isinstance(item, DataItem):
        item = item.delimiter
    if isinstance(item, DataType):
        return item.desc if isinstance(item.token_type, UniqueToken) else ''

def key_has_data(item: Union[int, Static]) -> bool:
    '''
    Check whether key has data.
    '''
    if isinstance(item, int):
        return False
    return item.data != []
