'''
Utility functions for the parsing of configuration objects.
'''

import os
from copy import deepcopy

from ._tokens import Token

def get_locations(token, root) -> list:
    '''
    Retrieve the locations of the image directories given a starting point in
    the config.
    
    - token: the token to search for
    - root: a dictionary representing the data filestructure
    
    Preconditions:
    - token must be a valid hash token defined in _tokens.py.
    - root is a subset of the dictionary of the dataset filestructure config.
    '''
    return _get_locations(token, root, [])

def _get_locations(token: int, root: dict, current_path: list) -> list:
    '''
    Helper method for get_locations(). 
    
    - token: the token to search for
    - root: a dictionary representing the data filestructure
    - current_path: the steps to take to reach the current working dir of root
    - locations: the list of locations to return containing the token
    '''
    locations = []
    for location in root:
        value = root[location]
        if isinstance(value, dict):
            locations = locations + _get_locations(
                    token,
                    value,
                    current_path + [location])
        else:
            if value == token:
                locations.append(current_path)
    return locations

def get_object(path, root) -> dict:
    '''
    Retrieves the object in the config.
    
    - path: the path to reach the object
    - root: a dictionary representing the data filestructure
    
    Preconditions: 
    - the path is valid, relative to root
    - root is a subset of the dictionary of the dataset filestructure config.
    '''
    curr = root
    for step in path:
        curr = curr[step]
    return curr

def assert_token(item: dict, key: str, tokens: list[int] | int) -> None:
    '''
    Assert that the specified key in root matches the token.
    
    - item: a subdictionary representing the data filestructure
    - key: the key in root (must be surface level)
    - token: the token to check for
    '''
    if not isinstance(tokens, list):
        tokens = [ tokens ]
    for token in tokens:
        if item[key] != token:
            _raise_invalid_format(path_repr(item['path']+[key]), tokens, item[key])

def assert_keys_exist(item: dict, keys: list[str] | str) -> None:
    '''
    Assert that the specified list of keys exist.
    
    - item: a subdictionary representing the data filestructure
    - keys: a list of valid keys (must be surface level)
    '''
    if not isinstance(keys, list):
        keys = [ keys ]
    for key in keys:
        if key not in item:
            raise ValueError(f'Required key {key} not found in {".".join(item["path"])}')

def assert_unique_key_exists(item: dict, keys: list[str]) -> None:
    '''
    Assert that the object contains at least one of keys.
    
    - item: a subdictionary representing the data filestructure
    - keys: a list of valid keys (must be surface level)
    '''
    if not any([key in item for key in keys]):
        raise ValueError(f'Object {path_repr(item["path"])} does not have any key of types \
{",".join(keys)}')
    if sum([key in item for key in keys]) > 1:
        raise ValueError(f'Object {path_repr(item["path"])} has too many keys of type \
{",".join(keys)} (required: 1)')

def path_repr(path: list[str]) -> str:
    '''
    Get the path representation, separated by periods.
    
    - path: the path to get repr of
    '''
    return '.'.join(path)

def _raise_invalid_format(path: list[str], poss_token: str | list, actual: int) -> None:
    '''
    Throw an error that the fields of an object are formatted incorrectly.
    
    - path: path to the object.
    - poss_token: the possible token(s), either one or a list of tokens.
    - actual: the actual token found.
    '''
    if isinstance(poss_token, list):
        raise ValueError(f'Object field {path_repr(path)} has invalid token \
{Token.get_type(actual)}, possible values: \
{",".join([Token.get_type(token_exp) for token_exp in poss_token])})')
    else:
        raise ValueError(f'Object field {path_repr(path)} has invalid token \
{Token.get_type(actual)}, possible values: {Token.get_type(poss_token)})')

def assert_dir_exists(base_dir: str, path: list[str]) -> None:
    '''
    Assert that the directory exists, given a path.
    
    - base_dir: the root of the dataset in the filesystem.
    - path: the direct folder path to the directory; this is not the same path
            as the path within the config.
    '''
    dir_path = os.path.join(base_dir, *path)
    if not os.path.exists(dir_path):
        raise ValueError(f'Invalid directory: {dir_path}')

def _expand_generic_item(root: dict) -> dict:
    '''
    Add items to the root, and pop item. Expand it based on whatever is available
    at the specified filepath.
    
    - root: a dictionary representing the file datastructure
    - dirpath: the existing path to the desired directory in the dataset
    
    Preconditions: 
    - root type is of Token.DIRECTORY.
    - 'item' exists in the path and 'items' does not.
    '''
    files = [name for name in os.listdir(root['path'])
             if not os.path.isdir(os.path.join(root['path'], name))]
    pattern = root.pop('item')
    folders = {}
    for file in files:
        folders[file] = deepcopy(pattern)
    root['items'] = folders

def _expand_generic_folder(root: dict) -> dict:
    '''
    Add folders to the root, and pop folder. Expand it based on whatever is available
    at the specified filepath.
    
    - root: a dictionary representing the file datastructure
    - dirpath: the existing path to the desired directory in the dataset
    
    Preconditions: 
    - root type is of Token.DIRECTORY.
    - 'folder' exists in the path and 'folders' does not.
    '''
    directories = [name for name in os.listdir(root['path'])
                   if os.path.isdir(os.path.join(root['path'], name))]
    pattern = root.pop('folder')
    folders = {}
    for directory in directories:
        folders[directory] = deepcopy(pattern)
    root['folders'] = folders

def expand_generics(root: dict, path) -> dict:
    '''
    Expand all generic types, and add path to each object.
    
    - root: a dictionary representing the file datastructure.
    - path: the current object path which represents root.
    '''



    if 'type' not in root:
        for item in root.keys():
            if not isinstance(root[item], dict):
                raise ValueError('Type token not found in object, but object is not a list of \
items/folders')
            root[item] = expand_generics(root[item], os.path.join(path, item))
        return root

    root['path'] = path
    if root['type'] == Token.DIRECTORY:
        if 'item' in root:
            _expand_generic_item(root)
            root['items'] = expand_generics(root['items'], path)
        if 'folder' in root:
            _expand_generic_folder(root)
            root['folders'] = expand_generics(root['folders'], path)

    return root
