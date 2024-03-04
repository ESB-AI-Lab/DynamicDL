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
import re
import os
import logging

from ._tokens import Token
from ._utils import assert_keys_exist

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

class TXT:
    '''
    Class for finding annotation information when stored in txt files.
    '''
    def __init__(self, root: dict, locations: list[str]) -> dict:
        '''
        Initialize the formats and detection methods.
        '''
        self.root = root
        self.style = root['format']
        assert_keys_exist(self.style, ['label', 'type', 'filename_structure', 'structure'])
        log.info('Verified surface level keys correct')
        
        self.image_set_identifier = None
        
        if str(Token.IMAGE_SET) in self.style['filename_structure']:
            log.info('Image set token detected in filename structure. Finding all image sets...')
            self.image_set_identifier = Token.FILENAME_IDENTIFIER
            self._get_image_sets(locations)

    def _get_image_sets(self, locations: list[list[str]]) -> list[str]:
        '''
        Find the image sets.
        
        - locations: the locations of the annotation files.
        '''
        match self.image_set_identifier:
            case Token.FILENAME_IDENTIFIER:
                pattern = self.style['filename_structure'].replace(str(Token.IMAGE_SET), "(.*)")
                for location in locations:
                    self.root['image_sets'].add(re.findall(pattern, location[-1])[0])