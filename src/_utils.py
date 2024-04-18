import json
from typing import Union
from itertools import repeat
from math import isclose

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

def _get_str(data):
    if isinstance(data, dict):
        return {str(key): _get_str(val) for key, val in data.items()}
    if isinstance(data, list):
        return [_get_str(val) for val in data]
    return str(data)

def get_str(data):
    '''Return pretty print string.'''
    return json.dumps(_get_str(data), indent=4).replace('"', '')

def check_map(it, num):
    return all(map(any, repeat(iter(it), num)))