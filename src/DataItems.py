'''
Represents all possible (required) data items for parsing a dataset.
'''

from typing import Self

from .Tokens.IdentifierTokens import IdentifierToken, StorageToken, WildcardToken, FilenameToken, \
                                     QuantityToken

class DataType:
    '''
    All possible data types.
    '''
    def __init__(self, desc: str, token_type: type[IdentifierToken]):
        self.desc: str = desc
        self.token_type: type[IdentifierToken] = token_type

    def __repr__(self) -> str:
        return f'<{self.desc}>'

    def __eq__(self, other: Self) -> bool:
        if self.__class__ != other.__class__:
            return False
        return self.desc == other.desc and self.token_type == other.token_type

    def verify_token(self, value: str) -> bool:
        '''
        Verify that a given value is valid for the datatype.
        '''
        return self.token_type.verify_token(value)

class DataTypes:
    '''
    Presets for DataType.
    '''
    IMAGE_SET: DataType = DataType('IMAGE_SET', StorageToken)
    IMAGE_SET_ID: DataType = DataType('IMAGE_SET_ID', StorageToken)
    ABSOLUTE_FILE: DataType = DataType('ABSOLUTE_FILE', FilenameToken)
    RELATIVE_FILE: DataType = DataType('RELATIVE_FILE', FilenameToken)
    IMAGE_NAME: DataType = DataType('IMAGE_NAME', StorageToken)
    IMAGE_ID: DataType = DataType('IMAGE_ID', StorageToken)
    CLASS_NAME: DataType = DataType('CLASS_NAME', StorageToken)
    CLASS_ID: DataType = DataType('CLASS_ID', StorageToken)
    XMIN: DataType = DataType('XMIN', QuantityToken)
    YMIN: DataType = DataType('YMIN', QuantityToken)
    XMAX: DataType = DataType('XMAX', QuantityToken)
    YMAX: DataType = DataType('YMAX', QuantityToken)
    WIDTH: DataType = DataType('WIDTH', QuantityToken)
    HEIGHT: DataType = DataType('HEIGHT', QuantityToken)
    GENERIC: DataType = DataType('GENERIC', WildcardToken)

class DataItem:
    '''
    Base, abstract class for representing a data item.
    '''
    def __init__(self, delimiter: DataType, value: str, insertion=False):
        if insertion:
            assert issubclass(delimiter.token_type, StorageToken), \
                   'Insertion can only valid if the dataclass supports storage'
            delimiter.token_type.add_token(value)
        assert delimiter.verify_token(value), 'Value is invalid for given delimiter type'
        self.delimiter: DataType = delimiter
        self.value: str = value
