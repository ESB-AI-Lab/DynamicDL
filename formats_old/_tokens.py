class Token:
    # modes
    CLASSIFICATION_MODE = hash('CLASSIFICATION_MODE')
    DETECTION_MODE = hash('DETECTION_MODE')
    SEGMENTATION_MODE = hash('SEGMENTATION_MODE')
    
    # main
    IMAGES = hash('IMAGES')
    ANNOTATION_FILE = hash('ANNOTATION_FILE')
    DATASET_ROOT = hash('DATASET_ROOT')
    DIRECTORY = hash('DIRECTORY')
    IMAGE = hash('IMAGE')
    IMAGE_CLASSIFIER = hash('IMAGE_CLASSIFIER')
    ANNOTATIONS = hash('ANNOTATIONS')
    TXT = hash('TXT')
    IMAGE_SET = hash('IMAGE_SET')
    BY_LINE = hash('BY_LINE')
    NONE = hash('NONE'),
    CLASS_IDX = hash('CLASS_IDX')
    GENERIC_ITEM = hash('GENERIC_ITEM')
    GENERIC_FOLDER = hash('GENERIC_FOLDER')
    ANNOTATION_FILE_STYLE = hash('ANNOTATION_FILE_STYLE')
    FILENAME_IDENTIFIER = hash('FILENAME_IDENTIFIER')
    
    # detection specific
    
    @staticmethod
    def get_type(token_hash) -> str:
        '''
        Get the string value of the token given its hash.
        '''
        members = [attr for attr in dir(Token) if not 
                   callable(getattr(Token, attr)) and not attr.startswith("__")]
        for token in members:
            if hash(token) == token_hash:
                return token
        raise ValueError('Invalid token.')
    
    @staticmethod
    def to_str(token_hash) -> str:
        '''
        Get the string version of the hash for searching.
        '''
        return str(token_hash)