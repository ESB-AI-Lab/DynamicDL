'''
ModeToken module
'''
from collections import namedtuple

from ..Token import Token

class ModeToken(Token):
    '''
    The ModeToken class supports 3 modes: classification, detection, and segmentation and represents
    the type of annotations that a dataset contains.
    '''
    Preset = namedtuple('Preset', ['CLASSIFICATION', 'DETECTION', 'SEGMENTATION'])
