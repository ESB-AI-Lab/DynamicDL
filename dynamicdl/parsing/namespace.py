
from typing import Union

from .static import Static
from .generic import Generic

class Namespace:
    '''
    The `Namespace` class functions as a collection of str, Static, and Generic objects which can
    all be viable values in some given text.
    
    :param names: Arguments to be provided which are valid str/Static/Generic objects that are all
        viable in the same key type.
    :type names: str | Static | Generic
    '''
    def __init__(self, *names: Union[str, Static, Generic]):
        self.names = names

    def match(self, entry: str):
        '''
        Return a list of the tokens' string values provided an entry string which follows the 
        pattern.
        
        :param entry: The entry string to be matched to the namespace patterns.
        :type entry: str
        :return: A boolean indicating success of the matching, and a list of the DataItems passed.
        :rtype: tuple[bool, list[DataItem]]
        '''
        for name in self.names:
            if isinstance(name, str):
                if entry == name:
                    return True, []
                continue
            if isinstance(name, Static):
                if name.name == entry:
                    return True, name.data
                continue
            res, data = name.match(entry)
            if res:
                return res, data
        return False, []
