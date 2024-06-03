from typing import Union, Any

from .._warnings import Warnings
from ..data.datatype import DataType
from .genericlist import GenericList
from .static import Static

class ImpliedList:
    '''
    The `ImpliedList` class is meant to pair its objects with their associated index. 
    Implied list, meant especially for pairing object. Allows the index of each item to be
    associated with that object. This is especially useful for datasets such as the YOLOv8 set,
    where list of class names are provided with their index inferred to be the class id.
    '''
    def __init__(self, form: Union[GenericList, list, Any], indexer: DataType, start: int = 0):
        self.form = form if isinstance(form, GenericList) else GenericList(form)
        self.indexer = indexer
        self.start = start
        self.length = len(self.form.form)

    def expand(
        self,
        path: list[str],
        dataset: list
    ) -> dict[Static, Any]:
        '''
        Expand implied list into dict of statics.
        
        :param dataset: The dataset data, which must be a list of values following some format.
        :type dataset: list
        :return: The parsed expansion of `Static` values, always a list. Single values are converted
            to lists of length 1. Note: for consistency lists are converted to dicts with int keys.
        :rtype: dict[int, Any]
        '''
        from .._main._engine import expand_generics
        if not isinstance(dataset, list):
            Warnings.error('incorrect_type', path=path, got=type(dataset))

        form = {self.indexer: self.form}
        dataset = {str(i + self.start): dataset[self.length * i : self.length * (i + 1)]
                   for i in range(len(dataset) // self.length)}
        return expand_generics(
            path,
            dataset,
            form
        )
