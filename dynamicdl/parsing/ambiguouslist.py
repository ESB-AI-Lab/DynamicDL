from typing import Union, Any, Optional
from tqdm import tqdm

from .._utils import union
from .genericlist import GenericList
from .static import Static

class AmbiguousList:
    '''
    Ambiguous List. Used to represent when an item could either be in a list, or a solo item.
    This is primarily used for XML files.
    
     - `form` (`GenericList | list | Any`): effectively wrapper for GenericList. Either provide with
        GenericList instantiation values or provide a GenericList.
    '''
    def __init__(self, form: Union[GenericList, list, Any]):
        self.form = form if isinstance(form, GenericList) else GenericList(form)

    def expand(
        self,
        path: list[str],
        dataset: Any,
        pbar: Optional[tqdm],
        depth: int = 0
    ) -> dict[Static, Any]:
        '''
        Expand potential list into dict of statics.
        
         - `dataset` (`list[Any]`): the dataset data, which should follow the syntax of 
            `DynamicData` data.
        '''
        dataset = union(dataset)
        return self.form.expand(
            path,
            dataset,
            pbar,
            depth=depth
        )
