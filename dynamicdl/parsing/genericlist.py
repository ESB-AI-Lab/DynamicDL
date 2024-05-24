
from typing import Union, Any, Optional
from tqdm import tqdm

from .._utils import union, load_config, Warnings
from .static import Static

config = load_config()

class GenericList:
    '''
    Generic list item. Items inside the list are expected to repeat mod `len(form)`.
    
     - `form` (`list[Any] | Any`): the form to stick to. Each entry in `form` must be some valid
        form following the syntax of `DynamicData` forms.
    '''
    def __init__(
        self,
        form: Union[list[Any], Any]
    ) -> None:
        self.form = union(form)

    def expand(
        self,
        path: list[str],
        dataset: list[Any],
        pbar: Optional[tqdm],
        depth: int = 0
    ) -> tuple[dict[Static, Any], list]:
        '''
        Expand list into dict of statics.
        
         - `dataset` (`list[Any]`): the dataset data, which should follow the syntax of 
            `DynamicData` data.
        '''
        from ..engine import expand_generics
        if depth >= config['MAX_PBAR_DEPTH']:
            pbar = None
        if pbar:
            pbar.set_description(f'Expanding generics: {"/".join(path)}')
        if len(dataset) % len(self.form) != 0:
            Warnings.error('generic_list_length', length1=len(dataset), length2=len(self.form))
        item_list: list[Any] = []
        item: list[Static | dict] = []
        pairings = []
        for index, entry in enumerate(dataset):
            result, pairing = expand_generics(
                path + [str(index)],
                entry,
                self.form[index % len(self.form)],
                pbar,
                depth = depth + 1
            )
            pairings += pairing
            item.append(result)
            if (index + 1) % len(self.form) == 0:
                item_list.append(dict(enumerate(item)))
                item = []
        return dict(enumerate(item_list)), pairings
