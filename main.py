'''
Main scripts for creating dataloaders. Primary command is get_dataset().
'''

import json
import torch


from formats._tokens import Token
from formats._utils import get_locations, get_object, assert_token, \
        assert_unique_key_exists, assert_keys_exist, expand_generics

def get_dataset(config: dict, base_dir: str) -> torch.utils.data.Dataset:
    '''
    Returns a dataset (torch.utils.data.Dataset) according to the config.
    
    - config: a dict object according to the specifications of the docs.
    - base_dir: the filepath to the dataset
    '''
    assert_keys_exist(config, ['label', 'type', 'format', 'root', 'formats'])
    # format = Token.get_type(config['format'])
    config['root'] = expand_generics(config['root'], base_dir)
    locations = _find_annotations(config['root'])
    return locations

def _find_annotations(root: dict) -> dict:
    '''
    Find the location of the annotations folder(s).
    
    - root: a dictionary representing the data filestructure
    - base_dir: the filepath to the dataset
    '''
    dirs = []
    annotation_locations = get_locations(Token.ANNOTATIONS, root)
    annotation_objects = [get_object(location, root) for location in annotation_locations]
    for items in annotation_objects:
        assert_unique_key_exists(items, ['items'])
        for item in items['items'].values():
            assert_token(item, 'type', Token.ANNOTATION_FILE)
            dirs.append(item['path'])
    return dirs

if __name__ == '__main__':
    dataset = {
        "label": Token.DATASET_ROOT,
        "type": Token.CLASSIFICATION_MODE,
        "format": Token.TXT,
        "root": {
            "images": {
                "label": Token.IMAGES,
                "type": Token.DIRECTORY,
                "item": {
                    "label": Token.GENERIC_ITEM,
                    "type": Token.IMAGE,
                    "filename": f"{Token.IMAGE_CLASSIFIER}.jpg"
                }
            },
            "annotations": {
                "label": Token.GENERIC_FOLDER,
                "type": Token.DIRECTORY,
                "folder": {
                    "label": Token.ANNOTATIONS,
                    "type": Token.DIRECTORY,
                    "item": {
                        "label": Token.GENERIC_ITEM,
                        "type": Token.ANNOTATION_FILE,
                        "format": "pets_format"
                    }
                }
            }
        },
        "formats": {
            "pets_format": {
                "label": Token.ANNOTATION_FILE,
                "type": Token.TXT,
                "filename_structure": f"{Token.IMAGE_SET}.txt",
                "structure": {
                    "line_offset": 0,
                    "type": Token.BY_LINE,
                    "line_structure": f"{Token.IMAGE_CLASSIFIER} {Token.NONE} {Token.NONE} {Token.CLASS_IDX}"
                }
            }
        }
    }
    print(json.dumps(get_dataset(dataset, '/Users/atong/Documents/Datasets/'), indent=4))