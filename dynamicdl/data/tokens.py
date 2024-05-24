import os
from typing import Any

from .._utils import union

class Token:
    '''
    The Token class is the base class which carries important information into Data objects for data
    parsing functions. Subclasses of this class may have specific requirements for content.
    
    All implementations of the Token class should not be static but also not use self, for
    compatibility reasons (may be changed in the future)
    '''
    def __init__(self) -> None:
        pass

    def verify_token(self, token: Any) -> bool:
        '''
        Checks whether the token is in valid format in accordance with the identifier. 
        
         - `token`: the token to check
        '''
        return token != ''

    def transform(self, token: Any) -> Any:
        '''
        Transform the token from a string value to token type.
        
         - `token`: the token to transform
        '''
        return token

class RedundantToken(Token):
    '''
    RedundantToken items are used for when a data item stores multiple values of itself per image
    or unique item. Cases like these include multiple bounding boxes or segmentation objects.
    '''
    def transform(self, token: str) -> Any:
        return union(token)

class UniqueToken(Token):
    '''
    UniqueToken items are used when an identifier is a unique item pertaining to any property of an
    image or entry. Unique tokens serve as valid IDs for identifying each data entry in the dataset.
    '''

class WildcardToken(Token):
    '''
    The WildcardToken class represents a generic wildcard which can stand for anything and will not 
    be used for any identifiers. The key difference is that these tokens are not affected by merge
    operations.
    '''

class WildcardWordToken(WildcardToken):
    '''
    Disallows spaces in the wildcard.
    '''
    def verify_token(self, token: str) -> bool:
        return super().verify_token(token) and any(filter(str.isspace, token))

class FilenameToken(UniqueToken):
    '''
    The FilenameToken class is a Token which checks for valid absolute filenames.
    '''
    def verify_token(self, token: Any) -> bool:
        return super().verify_token(token) and os.path.exists(token)

class IDToken(Token):
    '''
    Represents an ID. Items must be integers.
    '''
    def verify_token(self, token: Any) -> bool:
        return isinstance(token, int)

    def transform(self, token: str) -> Any:
        return int(token)

class WildcardIntToken(IDToken, WildcardToken):
    '''
    Wildcards for only integers.
    '''

class QuantityToken(Token):
    '''
    Represents a numeric quantity. Can be int or float.
    '''
    def verify_token(self, token: Any) -> bool:
        return isinstance(token, (int, float))

    def transform(self, token: str) -> Any:
        return float(token)

class WildcardQuantityToken(QuantityToken, WildcardToken):
    '''
    Wildcards for only quantities.
    '''

class RedundantQuantityToken(QuantityToken, RedundantToken):
    '''
    Represents a redundant numeric quantity.
    '''
    def verify_token(self, token: Any) -> bool:
        return isinstance(token, list) and all(isinstance(x, (float, int)) for x in token)

    def transform(self, token: str) -> Any:
        return list(map(float, union(token)))

class RedundantIDToken(IDToken, RedundantToken):
    '''
    Represents a redundant ID.
    '''
    def verify_token(self, token: Any) -> bool:
        return isinstance(token, list) and all(isinstance(x, int) for x in token)

    def transform(self, token: str) -> Any:
        return list(map(int, union(token)))

class RedundantObjectToken(RedundantToken):
    '''
    Represents a segmentation object.
    '''
    def verify_token(self, token: Any) -> bool:
        return (
            isinstance(token, list) and
            all(
                isinstance(x, list) and
                all(
                    isinstance(y, tuple) and
                    isinstance(y[0], float) and
                    isinstance(y[1], float)
                    for y in x
                )
                for x in token
            )
        )
    def transform(self, token: list) -> Any:
        if len(token) > 0 and isinstance(token[0], list):
            return token
        return [token]

class UniqueIDToken(IDToken, UniqueToken):
    '''
    Represents a unique ID.
    '''
