import re
from typing import Union

from .._warnings import Warnings
from ..data.datatype import DataType
from ..data.dataitem import DataItem
from .generic import Generic

class Alias:
    '''
    Class used when a placeholder in `Generic` could be interpreted multiple ways. For example, if
    `IMAGE_NAME` also contains `CLASS_NAME` and `IMAGE_ID`, we can extract all 3 tokens out using
    `Alias`. Counts for a single wildcard token (`{}`) when supplied in `Generic`. 
    
    Example:
    
    .. code-block:: python
    
        alias =  Alias([
            DataTypes.IMAGE_NAME,
            Generic('{}_{}', DataTypes.CLASS_NAME, DataTypes.IMAGE_ID)
        ])

    Now we can use `Generic(alias)` as a valid generic and it will obtain all contained DataTypes.
    
    :param generics: The list of Generic type objects which can be used for alias parsing.
    :type generics: list[Generic | DataType]
    :raises ValueError: There must be at least one item in the provided Generics
    '''
    def __init__(self, generics: list[Union['Generic', DataType]]) -> None:
        if len(generics) == 0:
            Warnings.error('generics_missing')
        self.generics = [generic if isinstance(generic, Generic)
                         else Generic('{}', generic) for generic in generics]
        self.patterns: list[str] = [generic.pattern for generic in self.generics]
        self.aliases: list[tuple[DataType, ...]] = [generic.data for generic in self.generics]
        self.desc = ''.join([token.desc for alias in self.aliases for token in alias])

    def match(self, entry: str) -> tuple[bool, list[DataItem]]:
        '''
        Return a list of DataItems if matched successfully. Used for internal processing functions.
        
        :param entry: The entry string to be matched to the alias pattern.
        :type entry: str
        :return: A boolean indicating success of the matching, and a list of the DataItems passed.
        :rtype: tuple[bool, list[DataItem]]
        '''
        result: list[DataItem] = []
        for pattern, alias in zip(self.patterns, self.aliases):
            pattern: str = pattern.replace('{}', '(.+)')
            matches: list[str] = re.findall(pattern, entry)
            try:
                if not matches:
                    return False, []
                # if multiple token matching, extract first matching; else do nothing
                if isinstance(matches[0], tuple):
                    matches = matches[0]
                for data_type, match in zip(alias, matches):
                    result.append(DataItem(data_type, match))
            except ValueError:
                return False, []
        return True, result

    def __repr__(self) -> str:
        return str(dict(zip(self.patterns, self.aliases)))
