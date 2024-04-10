from typing import Union

def union(item: Union[list[object], object]) -> list[object]:
    '''
    Returns the item in a list format if not already in a list format.
    '''
    if not isinstance(item, list):
        item = [ item ]
    return item

def next_avail_id(idx_to_name: dict[int, str]) -> int:
    for i, idx in enumerate(idx_to_name):
        if i != idx: return i
    return i + 1