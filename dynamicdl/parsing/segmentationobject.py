from typing import Union, Any, Optional
from tqdm import tqdm

from .._utils import load_config
from .._warnings import Warnings
from ..data.datatypes import DataTypes
from ..data.dataitem import DataItem
from ..data.dataentry import DataEntry
from .genericlist import GenericList
from .static import Static

config = load_config()

class SegmentationObject:
    '''
    Object to represent a collection of polygonal coordinates for segmentation. Functionally serves
    the purpose of being a wrapper class for `GenericList` and should be instantiated when the only
    contents inside are `DataTypes.X` and `DataTypes.Y` items as well as non-data items. This
    class therefore provides a way to bundle together POLYGON data types with variable length points
    for handling thereafter.
    
    :param form: Either a GenericList object or a list which will create a GL.
    :type form: GenericList | list
    '''
    def __init__(
        self,
        form: Union[GenericList, list]
    ) -> None:
        if isinstance(form, list):
            form = GenericList(form)
        self.form = form

    @staticmethod
    def _merge(
        data: Union[dict[Union[Static, int], Any], Static]
    ) -> DataEntry:
        # base cases
        if isinstance(data, Static):
            return DataEntry(data.data)
        recursive = []

        # get result
        for key, val in data.items():
            result = SegmentationObject._merge(val)
            # unique entry result
            if isinstance(result, DataEntry):
                if isinstance(key, Static):
                    result = DataEntry.merge(DataEntry(key.data), result)
                if result.unique:
                    recursive.append([result])
                else:
                    recursive.append(result)

        # if inside unique loop, either can merge all together or result has multiple entries
        result = DataEntry([])
        for item in recursive:
            result = DataEntry.merge(item, result)
        return result

    def expand(
        self,
        path: list[str],
        dataset: list[Any],
        pbar: Optional[tqdm],
        depth: int = 0
    ) -> tuple[dict[Static, Any], list]:
        '''
        Evaluate object by expanding and merging, and extracting the corresponding X, Y values
        which define the SegmentationObject.
        
        :param dataset: The dataset data, which should follow the syntax of `DynamicData` data.
        :type dataset: list[Any]
        '''
        if depth >= config['MAX_PBAR_DEPTH']:
            pbar = None
        if pbar:
            pbar.set_description(f'Expanding generics: {"/".join(path)}')
        item_dict, _ = self.form.expand(
            path,
            dataset,
            pbar,
            depth=depth
        )
        entry = self._merge(item_dict)
        entry.data.pop('GENERIC', '')
        x = entry.data.get('X').value
        y = entry.data.get('Y').value
        if x is None or y is None or len(entry.data) != 2:
            Warnings.error( 'invalid_seg_object', keys=", ".join(list(entry.data.keys())))
        if len(x) != len(y):
            Warnings.error(
                'row_mismatch',
                name1='X',
                name2='Y',
                len1=len(x),
                len2=len(y)
            )
        return Static('SegObject', DataItem(DataTypes.POLYGON, list(zip(x, y)))), []