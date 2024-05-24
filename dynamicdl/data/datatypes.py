from .tokens import Token, UniqueToken, UniqueIDToken, RedundantIDToken, RedundantObjectToken, \
                    RedundantQuantityToken, RedundantToken, WildcardIntToken, \
                    WildcardQuantityToken, WildcardToken, WildcardWordToken, FilenameToken, IDToken
from .datatype import DataType

class DataTypes:
    '''
    The `DataTypes` class contains static presets for `DataType` types. Below is a description of
    all presets currently available:
    '''

    # main types
    IMAGE_SET_NAME = DataType(
        'IMAGE_SET_NAME',
        RedundantToken(),
        doc = 'Represents the name of an image set. This includes any valid strings, but is not '
            'meant to store the ID of the image set; see `IMAGE_SET_ID`. Image sets are used to '
            'allocate specific entries to a group which can be split when dataloading. Most '
            'commonly, image set names will be `train`, `val`, or `test`. [GENERAL]'
    )
    IMAGE_SET_ID = DataType(
        'IMAGE_SET_ID',
        RedundantIDToken(),
        doc = 'Represents the ID of an image set. This includes any valid integers. The named '
            'complement of this DataType is `IMAGE_SET_NAME`. See above for details. [GENERAL]'
    )
    ABSOLUTE_FILE = DataType(
        'ABSOLUTE_FILE',
        FilenameToken(),
        doc = 'Represents the **absolute** filepath of an entry image only. This DataType is '
            'automatically generated in `Image` and `File` type objects when parsing, but can also '
            'be used to parse data. All valid values under `ABSOLUTE_FILE` must be a valid '
            'filepath on the user\'s filesystem. `RELATIVE_FILE` is currently not supported, but '
            'may be in future versions. [GENERAL]'
    )
    ABSOLUTE_FILE_SEG = DataType(
        'ABSOLUTE_FILE_SEG',
        FilenameToken(),
        doc = 'Represents the **absolute** filepath of an entry segmentation mask only. This '
            'DataType is also automatically generated in `Image` and `File` type objects when '
            'parsing, but can also be used to parse data. All valid values under `ABSOLUTE_FILE` '
            'must be a valid filepath on the user\'s filesystem. `RELATIVE_FILE_SEG` is currently '
            'not supported, but may be in future versions. [GENERAL]'
    )
    IMAGE_NAME = DataType(
        'IMAGE_NAME',
        UniqueToken(),
        doc = 'Represents an identifier token for image entries via a string description. As of '
            '0.1.1-alpha all `IMAGE_NAME` entries must be unique as it serves as a sole identifier '
            'for image entries. Accepts parsed strings. Its ID complement can be found under '
            '`IMAGE_ID`. [GENERAL]'
    )
    IMAGE_ID = DataType(
        'IMAGE_ID',
        UniqueIDToken(),
        doc = 'The ID (parsed to int) complement for `IMAGE_NAME`. Behaves just like its '
            'complement. [GENERAL]'
    )
    GENERIC = DataType(
        'GENERIC',
        WildcardToken(),
        doc = 'A generic token with no significance that can be used as a wildcard token for '
            'parsing. Can represent anything, and any type. [GENERAL]'
    )
    GENERIC_INT = DataType(
        'GENERIC_INT',
        WildcardIntToken(),
        doc = 'Same as `GENERIC`, except accepts only integer types. [GENERAL]'
    )
    GENERIC_QUANTITY = DataType(
        'GENERIC_QUANTITY',
        WildcardQuantityToken(),
        doc = 'Same as `GENERIC`, except accepts only numeric types (i.e. float and int). [GENERAL]'
    )
    GENERIC_WORD = DataType(
        'GENERIC_WORD',
        WildcardWordToken(),
        doc = 'Same as `GENERIC`, except accepts only one word, i.e. no spaces allowed. [GENERAL]'
    )

    # classification
    CLASS_NAME = DataType(
        'CLASS_NAME',
        Token(),
        doc = 'Represents the classification class name of an image entry. There can only be one '
            'class per image entry, and accepts parsed strings. Its ID complement can be found '
            'under `CLASS_ID`. [CLASSIFICATION]'
    )
    CLASS_ID = DataType(
        'CLASS_ID',
        IDToken(),
        doc = 'The ID (parsed to int) complement for `CLASS_NAME`. Behaves just like its '
            'complement. [CLASSIFICATION]'
    )

    # detection
    BBOX_CLASS_NAME = DataType(
        'BBOX_CLASS_NAME',
        RedundantToken(),
        doc = 'Represents the detection class name of an image entry. There can be multiple '
            'classes per image entry, and accepts parsed strings. Its ID complement can be found '
            'under `BBOX_CLASS_ID`. Each detection class must have a one-to-one correspondence to '
            'a valid bounding box when in the same hierarchy. When in different hierarchies it, '
            'just like other redundant types, will expand naturally to fit the existing length. '
            '[DETECTION]'
    )
    BBOX_CLASS_ID = DataType(
        'BBOX_CLASS_ID',
        RedundantIDToken(),
        doc = 'The ID (parsed to int) complement for `BBOX_CLASS_NAME`. Behaves just like its '
            'complement. [DETECTION]'
    )
    XMIN = DataType(
        'XMIN',
        RedundantQuantityToken(),
        doc = 'The minimum x-coordinate in the bounding box. Must be accompanied with `YMIN` or '
            'else has no effect, and must be accompanied either with `XMAX` or `WIDTH` and their '
            'y-counterparts. [DETECTION]'
    )
    YMIN = DataType(
        'YMIN',
        RedundantQuantityToken(),
        doc = 'The minimum y-coordinate in the bounding box. Must be accompanied with `XMIN` or '
            'else has no effect, and must be accompanied either with `YMAX` or `HEIGHT` and their '
            'y-counterparts. [DETECTION]'
    )
    XMAX = DataType(
        'XMAX',
        RedundantQuantityToken(),
        doc = 'The maximum x-coordinate in the bounding box. Must be accompanied with `YMAX` or '
            'else has no effect, and must be accompanied either with `XMIN` or `WIDTH` and their '
            'y-counterparts. [DETECTION]'
    )
    YMAX = DataType(
        'YMAX',
        RedundantQuantityToken(),
        doc = 'The maximum y-coordinate in the bounding box. Must be accompanied with `XMAX` or '
            'else has no effect, and must be accompanied either with `YMIN` or `HEIGHT` and their '
            'y-counterparts. [DETECTION]'
    )
    XMID = DataType(
        'XMID',
        RedundantQuantityToken(),
        doc = 'The midpoint x-coordinate in the bounding box. Used to denote the vertical center '
            'of the bounding box. Must be accompanied with `YMID` to define a central point, and '
            'with either `XMIN` or `XMAX` to fill the bounding box. [DETECTION]'
    )
    YMID = DataType(
        'YMID',
        RedundantQuantityToken(),
        doc = 'The midpoint y-coordinate in the bounding box. Used to denote the vertical center '
            'of the bounding box. Must be accompanied with `XMID` to define a central point, and '
            'with either `YMIN` or `YMAX` to fill the bounding box. [DETECTION]'
    )
    X1 = DataType(
        'X1',
        RedundantQuantityToken(),
        doc = 'A bounding box x-coordinate. Can be in any order as long as it forms a valid '
            'bounding box with `X2`, `Y1`, and `Y2`. [DETECTION]'
    )
    Y1 = DataType(
        'Y1',
        RedundantQuantityToken(),
        doc = 'A bounding box y-coordinate. Can be in any order as long as it forms a valid '
            'bounding box with `X1`, `X2`, and `Y2`. [DETECTION]'
    )
    X2 = DataType(
        'X2',
        RedundantQuantityToken(),
        doc = 'A bounding box x-coordinate. Can be in any order as long as it forms a valid '
            'bounding box with `X1`, `Y1`, and `Y2`. [DETECTION]'
    )
    Y2 = DataType(
        'Y2',
        RedundantQuantityToken(),
        doc = 'A bounding box y-coordinate. Can be in any order as long as it forms a valid '
            'bounding box with `X1`, `X2`, and `Y1`. [DETECTION]'
    )
    WIDTH = DataType(
        'WIDTH',
        RedundantQuantityToken(),
        doc = 'The width of the bounding box. Must be accompanied with `HEIGHT` or else has no '
            'effect. Can be used as an alternative to defining `XMAX` or `XMIN`. [DETECTION]'
    )
    HEIGHT = DataType(
        'HEIGHT',
        RedundantQuantityToken(),
        doc = 'The height of the bounding box. Must be accompanied with `WIDTH` or else has no '
            'effect. Can be used as an alternative to defining `YMAX` or `YMIN`. [DETECTION]'
    )

    # segmentation
    SEG_CLASS_NAME = DataType(
        'SEG_CLASS_NAME',
        RedundantToken(),
        doc = 'Represents the segmentation class name of an image entry. There can be multiple '
            'classes per image entry, and accepts parsed strings. Its ID complement can be found '
            'under `SEG_CLASS_ID`. Each detection class must have a one-to-one correspondence to a '
            'valid bounding box when in the same hierarchy. When in different hierarchies it, just '
            'like other redundant types, will expand naturally to fit the existing length. '
            '[SEGMENTATION]'
    )
    SEG_CLASS_ID = DataType(
        'SEG_CLASS_ID',
        RedundantIDToken(),
        doc = 'The ID (parsed to int) complement for `SEG_CLASS_NAME`. Behaves just like its '
            'complement. [SEGMENTATION]'
    )
    X = DataType(
        'X',
        RedundantQuantityToken(),
        doc = 'A segmentation polygon x-coordinate. Used to define the vertices of a polygon for '
            'segmentation tasks. Each `X` coordinate must be paired with a corresponding `Y` '
            'coordinate to form a valid vertex. [SEGMENTATION]'
    )
    Y = DataType(
        'Y',
        RedundantQuantityToken(),
        doc = 'A segmentation polygon y-coordinate. Used to define the vertices of a polygon for '
            'segmentation tasks. Each `Y` coordinate must be paired with a corresponding `X` '
            'coordinate to form a valid vertex. [SEGMENTATION]'
    )
    POLYGON = DataType(
        'POLYGON',
        RedundantObjectToken(),
        doc = 'Should not be instantiated by the user as there is no way to parse it. However, it '
            'is automatically created upon every `SegmentationObject` wrapper of `X` and `Y` '
            'objects. This DataType is used internally for parsing. [SEGMENTATION]'
    )
