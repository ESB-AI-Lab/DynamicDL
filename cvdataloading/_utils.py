'''
Private utility functions for CVData.
'''
import json
from typing import Union, NoReturn
from itertools import repeat

def union(item: Union[list[object], object]) -> list[object]:
    '''
    Returns the item in a list format if not already in a list format.
    '''
    if not isinstance(item, list):
        item = [ item ]
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
    return all(map(any, repeat(iter(it), num)))

class MergeError(Exception):
    '''
    Exceptions raised while merging.
    '''

class LengthMismatchError(Exception):
    '''
    Exceptions raised when paired values have different lengths.
    '''

class Warnings:
    '''
    Predefined warning and error messages for CVData.
    '''
    # warnings
    file_ext = ('Warning: pattern has a . in it. If this is a file extension, omit in the pattern '
                'and include with keyword extensions. Disable this message with '
                'disable_warnings=True.')
    unsafe_load = ('Warning: attempting to load unsafe object (form). If you do not need the '
                   'original dataset form you should save the dataset with `safe=True`.')
    unsafe_save = ('Warning: attempting to save unsafe object (form). If you do not need to '
                   're-parse the dataset format again you should save with `safe=True`.')
    

    # errors
    merge_conflict = ('Conflicting information found while merging two entries:'
                      '\n - {first}\n - {second}', MergeError)
    merge_unique_conflict = ('Conflicting unique information found while applying data to entry:'
                             '\nParent unique ID {parent} and data unique ID {token} do not match',
                             MergeError)
    merge_redundant_conflict = ('Illegal differences ({overlap}) in more than one redundant group:'
                                '\n - {first}\n - {second}', MergeError)
    duplicate_images = ('Found equivalent md5-hash images in the dataset with the following image '
                        'names:{duplicates}', ValueError)
    invalid_scale = ('Invalid bounding box scale option provided ({scale}). Valid options are '
                     '\'zeroone\', \'full\', or \'auto\'.', ValueError)
    invalid_scale_data_bbox = ('Found negative values in provided coordinates at idx {id}. If you '
                               'are using WIDTH/HEIGHT calculations, you may have incorrectly used '
                               'XMID/YMID/XMAX/YMAX when it should be another value. [-1, 1] scale '
                               'is not currently accepted. TODO', ValueError)
    invalid_scale_data = ('Found negative values in provided coordinates at idx {id}. [-1, 1] '
                          'scale is not currently accepted. TODO', ValueError)
    invalid_id_map = ('Invalid {type} id {i} assigned to name {v}, expected {expect}.', KeyError)
    row_mismatch = ('Paired values {name1} and {name2} have different lengths {len1} and {len2}.',
                    LengthMismatchError)
    incomplete_bbox = ('Incomplete bounding box data! Bounding box columns were detected but could '
                       'not find a valid configuration. Found columns {columns} which are '
                       'incomplete.', KeyError)
    invalid_seg_object = ('Found invalid SegmentationObject instance with datatypes that has been '
                          'parsed as: {keys}.\nOnly \'X\' and \'Y\' values allowed.', KeyError)
    already_parsed = ('Dataset has already been parsed. Use override=True to override',
                      RuntimeError)
    image_set_missing = ('Image set {imgset_name} {image_set} doesn\'t exist!.', KeyError)
    image_set_empty = ('Image set {image_set} is empty after cleanup.', KeyError)
    mode_unavailable = ('Desired mode {mode} not available.', KeyError)
    nan_exists = ('Found NaN values that will cause errors in row: \n{row}. You can automatically '
                  'purge NaN values with remove_invalid=True.', ValueError)
    new_exists = ('New set {type} {imgset_name} already exists!', KeyError)
    split_invalid = ('Split fraction invalid, cannot be {desc} 1', ValueError)
    file_exists = ('File already exists: {filename}. Set overwrite=True to overwrite the file.',
                   OSError)
    is_none = ('{desc} cannot be None!', TypeError)
    fail_generic_match = ('Failed to match: {dataset} to {root}', MergeError)
    static_missing = ('Static value {value} not found in dataset', ValueError)
    inappropriate_type = ('Inappropriate type found for value {value}', TypeError)
    nonredundant_add = ('Cannot add to item which is not redundant', ValueError)
    data_invalid = ('Value {value} is invalid for given delimiter type {delimiter}', ValueError)
    generics_missing = ('Must have at least 1 generic in alias instance.', ValueError)
    generic_list_length = ('List length ({length1}) must be a multiple of length of provided form '
                           '({length2})', LengthMismatchError)
    invalid_dataset = ('This may not be a valid CVDataset, or the dataset is too old.', ValueError)
    pairings_missing = ('Must have at least 2 pairing datatypes.', ValueError)
    invalid_pairing = ('Pairings are not of the same type, either must all be redundant or none'
                       'redundant. Given: {paired}', MergeError)
    invalid_shape = ('{mode} requires equal input image dimensions, try using the `resize` '
                     'parameter when getting dataloader, or specify your own collate function.',
                     RuntimeError)
    empty_bbox = ('Bounding box is empty for image at {file}.', RuntimeError)

    @staticmethod
    def warn(name: str, **kwargs: str) -> None:
        '''
        Warn from one of the warning presets.
        '''
        print(getattr(Warnings, name).format(**kwargs))

    @staticmethod
    def error(name: str, **kwargs: str) -> NoReturn:
        '''
        Throw an error from one of the error presets.
        '''
        error = getattr(Warnings, name)
        exception = error[1]
        raise exception(error[0].format(**kwargs))
