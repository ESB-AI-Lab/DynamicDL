'''
ModeToken module.
'''

from typing import Self

from .Token import Token

class ModeToken(Token):
    '''
    The ModeToken class supports 3 modes: classification, detection, and segmentation and represents
    the type of annotations that a dataset contains.
    '''
    def __init__(self, mode: str):
        self.mode = mode

    @classmethod
    def classification(cls) -> Self:
        '''
        Initialize a classification mode token.
        '''
        return cls('Classification')

    @classmethod
    def detection(cls) -> Self:
        '''
        Initialize a classification mode token.
        '''
        return cls('Detection')

    @classmethod
    def segmentation(cls) -> Self:
        '''
        Initialize a classification mode token.
        '''
        return cls('Segmentation')

    def __repr__(self) -> str:
        return self.mode
