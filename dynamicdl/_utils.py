'''
Private utility functions for DynamicData.
'''
import json
from typing import Union

from .config import config

def union(item: Union[list[object], object]) -> list[object]:
    '''
    Returns the item in a list format if not already in a list format.
    '''
    if not isinstance(item, list):
        item = [item]
    return item

def next_avail_id(idx_to_name: dict[int, str]) -> int:
    '''Get next available id and fills in gaps'''
    for i, idx in enumerate(idx_to_name):
        if i != idx:
            return i
    return len(idx_to_name)

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
    '''Check an iterator for num occurrences'''
    return sum(it) >= num

def load_config():
    '''Load config object'''
    return config

def key_has_data(item) -> bool:
    '''
    Check whether key has data.
    '''
    if isinstance(item, int):
        return False
    return item.data != []
