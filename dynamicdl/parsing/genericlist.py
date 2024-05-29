from typing import Union, Any

from .._utils import union, load_config
from .._warnings import Warnings
from .static import Static

config = load_config()

class GenericList:
    '''
    Generic list item. Items inside the list are expected to repeat mod `len(form)`.
    
    Example:
    
    .. code-block:: json

        {
            "bounding_box": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0
            ]
        }
        
    Suppose that we wish to parse the bounding boxes for this particular json file. Let each value
    represent X1, Y1, X2, Y2 as needed. Then we can parse the form as follows:
    
    .. code-block:: python

        form = {
            "bounding_box": [
                DataTypes.X1,
                DataTypes.Y1,
                DataTypes.X2,
                DataTypes.Y2
            ]
        }
        
    Suppose the format was changed to x, y pairs:
    
    .. code-block:: json

        {
            "bounding_box": [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0]
            ]
        }
    
    Its corresponding form:
    
    .. code-block:: python

        form = {
            "bounding_box": [
                [DataTypes.X1, DataTypes.Y1],
                [DataTypes.X2, DataTypes.Y2]
            ]
        }
        
    During parsing, the standard python list is always inferred to be a `GenericList`. When the list
    items are 1:1, `GenericList` parses properly regardless.
    
    :param form: The form to stick to. Each entry in `form` must be some valid generic-like form,
        and all items inside the `form` list will be combined into one object upon parsing.
        Further lines in the list are expected to conform to the same scheme as the first entry.
    :type form: list[Any] | Any
    '''
    def __init__(
        self,
        form: Union[list[Any], Any]
    ) -> None:
        self.form = union(form)

    def expand(
        self,
        path: list[str],
        dataset: list[Any]
    ) -> tuple[dict[Static, Any], list]:
        '''
        Expand list into dict of statics, for internal processing.
        
        :param dataset: The dataset data, which should follow the syntax of `DynamicData` data.
        :type dataset: list[Any]
        '''
        from .._main._engine import expand_generics
        if len(dataset) % len(self.form) != 0:
            Warnings.error('generic_list_length', length1=len(dataset), length2=len(self.form))
        item_list: list[Any] = []
        item: list[Static | dict] = []
        pairings = []
        for index, entry in enumerate(dataset):
            result, pairing = expand_generics(
                path + [str(index)],
                entry,
                self.form[index % len(self.form)]
            )
            pairings += pairing
            item.append(result)
            if (index + 1) % len(self.form) == 0:
                item_list.append(dict(enumerate(item)))
                item = []
        return dict(enumerate(item_list)), pairings
