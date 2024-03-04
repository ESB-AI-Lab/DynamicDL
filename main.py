'''
Main scripts for creating dataloaders. Primary command is get_dataset().
'''

import os
import logging
import torch

from formats_old._tokens import Token
from formats_old.txt import TXT
from formats_old._utils import get_locations, get_object, \
        assert_unique_key_exists, assert_keys_exist, expand_generics

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


def get_dataset(config: dict) -> torch.utils.data.Dataset:
    '''
    Returns a dataset (torch.utils.data.Dataset) according to the config.
    
    - config: a dict object according to the specifications of the docs.
    '''
    assert_keys_exist(config, ['label', 'type', 'root', 'root_dir', 'format'])
    log.info('Verified surface level keys correct')
    style = Token.get_type(config['format']['type'])
    log.info('Found style of annotation files: %s', style)

    config['root'] = expand_generics(config['root'], config['root_dir'])
    log.info('Expanded generic items and folders.')

    locations = _find_annotations(config['root'])
    log.info('Found all annotation file locations.')

    # log.info(json.dumps(config, indent=4))

    if 'image_sets' not in config:
        log.info('Image sets not detected in config. Initializing and detecting...')
        config['image_sets'] = set()

    match style:
        case 'TXT':
            parser = TXT(config, locations)
        case _:
            raise ValueError(f'Style token not valid: {style}')
    
    log.info('Found image sets %s', config['image_sets'])
        
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
    for path, items in zip(annotation_locations, annotation_objects):
        assert_unique_key_exists(items, ['items'])
        files = items['items']
        for file in files.keys():
            dirs.append(path + [file])
            log.info('Located annotation file %s', path + [file])
    return dirs

if __name__ == '__main__':
    dataset = {
        "label": Token.DATASET_ROOT,
        "type": Token.CLASSIFICATION_MODE,
        "root_dir": "/Users/atong/Documents/Datasets",
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
                        "type": Token.ANNOTATION_FILE
                    }
                }
            }
        },
        "format": {
            "label": Token.ANNOTATION_FILE_STYLE,
            "type": Token.TXT,
            "filename_structure": f"{Token.IMAGE_SET}.txt",
            "structure": {
                "line_offset": 0,
                "type": Token.BY_LINE,
                "line_structure": f"{Token.IMAGE_CLASSIFIER} {Token.NONE} {Token.NONE} {Token.CLASS_IDX}"
            }
        }
    }
    get_dataset(dataset)
