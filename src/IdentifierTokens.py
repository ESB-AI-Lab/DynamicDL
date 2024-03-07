'''
Identifier Token type module
'''

import os
from abc import abstractmethod

from .Token import Token

class IdentifierToken(Token):
    '''
    The IdentifierToken class is an abstract class which carries important information into 
    StringFormatTokens for data parsing functions. Subclasses of this class may have specific 
    requirements for content.
    '''
    def __init__(self):
        pass

    @abstractmethod
    def verify_token(self, token: str) -> bool:
        '''
        Checks whether the token is in valid format in accordance with the identifier.
        
        - token: the token to check
        '''

class StorageToken(IdentifierToken):
    '''
    The StorageToken class possesses a set of elements which checks upon itself for membership.
    '''
    def __init__(self):
        self.items: set[str] = set()

    def verify_token(self, token: str, insertion: bool = False) -> bool:
        if insertion:
            self.items.add(token)
        return token in self.items

class RedundantStorageToken(StorageToken):
    '''
    The RedundantStorageToken class possesses a set of elements which checks upon itself, but also
    allows for a data item to store a list of items instead of just one value.
    '''

class UniqueToken(IdentifierToken):
    '''
    The UniqueToken class possesses a set of elements which checks upon itself for membership.
    '''
    def __init__(self):
        self.items: set[str] = set()

    def verify_token(self, token: str, insertion: bool = False) -> bool:
        if insertion:
            self.items.add(token)
        return token in self.items

class WildcardToken(IdentifierToken):
    '''
    The WildcardToken class represents a generic wildcard which can stand for anything and will not 
    be used for any identifiers.
    '''

    def verify_token(self, token: str) -> bool:
        '''
        Any string passes the wildcard check. Dummy method for assertions.
        
        - token (str): the token parsed from StringFormatToken.match()
        '''
        return True

class FilenameToken(UniqueToken):
    '''
    The FilenameToken class is an IdentifierToken which checks for valid filenames.
    '''
    def verify_token(self, token: str, insertion: bool = True) -> bool:
        '''
        Any proper filename passes the check assuming it exists.
        
        - root (str): the root to the main dataset directory.
        - token (str): the token parsed from StringFormatToken.match()
        '''
        return os.path.exists(token) and super().verify_token(token, insertion=insertion)

class QuantityToken(IdentifierToken):
    '''
    Represents a numeric quantity.
    '''
    def verify_token(self, token: str) -> bool:
        '''
        Any proper filename passes the check assuming it exists.
        
        - token (str): the token parsed from StringFormatToken.match()
        '''
        return token.isnumeric()
