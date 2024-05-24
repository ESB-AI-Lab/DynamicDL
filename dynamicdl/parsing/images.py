'''
Image dummy classes.
'''

class ImageEntry:
    '''
    Arbitrary image file to be used as a value in the key-value pairing of DynamicDL filestructure
    formats. It is a dummy object which provides absolute file and image data during processing,
    and is a marker object to recognize the presence of an image.
    '''
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "ImageEntry"

class SegmentationImage:
    '''
    Arbitrary segmentation image file to be used as a value in the key-value pairing of DynamicDL
    filestructure formats. It is a dummy object which provides absolute file and segmentation image
    map data during processing, and is a marker object to recognize the presence of an image.
    '''
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "SegmentationImage"
