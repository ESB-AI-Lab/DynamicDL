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
    @staticmethod
    @abstractmethod
    def verify_token(token: str) -> bool:
        '''
        Checks whether the token is in valid format in accordance with the identifier.
        
        - token: the token to check
        '''

class StorageToken(IdentifierToken):
    '''
    The StorageToken class possesses a set of elements which checks upon itself for membership.
    '''
    items = set()

    @staticmethod
    def verify_token(token: str) -> bool:
        '''
        Checks whether the token exists in the item pool.
        
        - token (str): the token to check for.
        '''
        return token in StorageToken.items

    @staticmethod
    def add_token(token: str) -> bool:
        '''
        Add the token to the pool of items. Returns false if it already exists.
        
        - token (str): the token to add.
        '''
        if token in StorageToken.items:
            return False
        StorageToken.items.add(token)
        return True

class WildcardToken(IdentifierToken):
    '''
    The WildcardToken class represents a generic wildcard which can stand for anything and will not 
    be used for any identifiers.
    '''

    @staticmethod
    def verify_token(token: str) -> bool:
        '''
        Any string passes the wildcard check. Dummy method for assertions.
        
        - token (str): the token parsed from StringFormatToken.match()
        '''
        return True

class FilenameToken(IdentifierToken):
    '''
    The FilenameToken class is an IdentifierToken which checks for valid filenames.
    '''
    @staticmethod
    def verify_token(token: str) -> bool:
        '''
        Any proper filename passes the check assuming it exists.
        
        - root (str): the root to the main dataset directory.
        - token (str): the token parsed from StringFormatToken.match()
        '''
        return os.path.exists(token)

class QuantityToken(IdentifierToken):
    '''
    Represents a numeric quantity.
    '''
    @staticmethod
    def verify_token(token: str) -> bool:
        '''
        Any proper filename passes the check assuming it exists.
        
        - token (str): the token parsed from StringFormatToken.match()
        '''
        return token.isnumeric()
