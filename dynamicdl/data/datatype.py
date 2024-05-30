from typing import Optional
from typing_extensions import Self

from .._warnings import Warnings
from .tokens import Token

class DataType:
    '''
    `DataType` is a container class for storing relevant dataset items. Token type options can be
    found in the `tokens` module. Warning: DataType instantiates are persistent through program
    execution, and can be accessed at the static dict `DataType.types`.

    :param desc: The purpose of the DataType. This should be unique for every new object.
    :type desc: str
    :param token_type: The token type of the DataType.
    :type token_type: Token
    '''
    types: dict[str, Self] = {}

    def __init__(self, desc: str, token_type: Token, doc: Optional[str] = None) -> None:
        self.desc: str = desc
        self.token_type: Token = token_type
        self.doc: str = doc if doc is not None else desc
        if desc in DataType.types:
            Warnings.error('type_exists', desc=desc)
        DataType.types[desc] = self

    def __repr__(self) -> str:
        return f'{self.doc}'

    def __eq__(self, other: Self) -> bool:
        if self.__class__ != other.__class__:
            return False
        return self.desc == other.desc

    def __hash__(self) -> int:
        return hash(self.desc)

    def verify_token(self, value: str) -> bool:
        '''
        Verify that a given value is valid for the datatype. Calls on internal Token
        functions for validation.
        
        - `value` (`str`): the value to check if it is compatible with the DataType.
        '''
        return self.token_type.verify_token(value)
