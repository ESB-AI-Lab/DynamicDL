'''
Oxford-IIIT Pets Dataset example:

format = {
    "label": Token.DATASET_ROOT,
    "type": Token.CLASSIFICATION_MODE,
    "root": {
        "images": {
            "label": Token.IMAGES,
            "type": Token.DIRECTORY,
            "item": {
                "label": Token.IMAGE,
                "type": Token.IMAGE,
                "filename": f"{Token.IMAGE_CLASSIFIER}.jpg"
            }
        }
        "annotations": {
            "label": Token.ANNOTATIONS,
            "type": Token.DIRECTORY,
            "item": [
                {
                    "label": Token.ANNOTATION_FILE,
                    "type": Token.TXT,
                    "format": "pets_format"
                }
            ]
        }
    },
    "formats": {
        "pets_format": {
            "label": Token.ANNOTATION_FILE,
            "type": Token.TXT,
            "filename_structure": f"{Token.IMAGE_SET}.txt"
            "structure": {
                "label": Token.
                "line_offset": 0,
                "type": Token.BY_LINE,
                "line_structure": f"{Token.IMAGE_CLASSIFIER} {Token.NONE} {Token.NONE} {Token.CLASS_IDX}"
            }
        }
    }
}
'''


import os
import torch

from ._tokens import Token

from ._utils import get_locations, get_object

class TXT:
    pass