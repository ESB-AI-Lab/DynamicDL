from typing import Any, Optional, Union
from tqdm import tqdm

from .._utils import load_config
from .._warnings import Warnings
from ..data.tokens import RedundantToken
from ..data.datatype import DataType
from ..data.datatypes import DataTypes
from ..data.dataitem import DataItem
from ..data.dataentry import DataEntry
from .static import Static

config = load_config()

class Pairing:
    '''
    Pairing is a wrapper class used to specify when two or more nonunique datatypes should be
    associated together. Most commonly used to pair ID and name together. 

    - `form` (`Any`): Whatever follows the DynamicData specified form as required. Pairing is a
        wrapper class so let it behave as it should
    - `paired` (`DataType`): Items which should be associated together.

    '''
    def __init__(self, form: Any, *paired: DataType) -> None:
        if len(paired) <= 1:
            Warnings.error('pairings_missing')
        self.paired = set(paired)
        self.paired_desc = {pair.desc for pair in paired}
        self.form = form
        self.redundant = isinstance(paired[0].token_type, RedundantToken)
        if self.redundant:
            if any(not isinstance(pair.token_type, RedundantToken) for pair in self.paired):
                Warnings.error('invalid_pairing', paired=', '.join(map(str, paired)))
        else:
            if any(isinstance(pair.token_type, RedundantToken) for pair in self.paired):
                Warnings.error('invalid_pairing', paired=', '.join(map(str, paired)))

    def update_pairing(self, entry: DataEntry) -> None:
        '''
        Update a data entry with pairing values, and does nothing if the pairing does not aplpy.
        
         - `entry` (`DataEntry`): the entry to apply this pairing
        '''
        entry_vals = set(entry.data.keys())
        overlap = entry_vals.intersection(self.paired_desc)
        if not overlap:
            return
        to_fill = self.paired_desc - overlap
        overlap = list(overlap)
        if not self.redundant:
            index = -1
            for i, pairing in enumerate(self.pairs):
                if entry.data[overlap[0]].value == pairing.data[overlap[0]].value:
                    index = i
                    break
            if index == -1:
                return
            for check in overlap:
                res = pairing.data.get(check, {}).value == entry.data[check].value
                if not res:
                    return
            for empty in to_fill:
                entry.data[empty] = DataItem.copy(pairing.data[empty])
            return
        indices: list[Optional[int]] = []
        for v in entry.data[overlap[0]].value:
            index = -1
            for i, pairing in enumerate(self.pairs):
                if v == pairing.data[overlap[0]].value:
                    index = i
                    break
            if index == -1:
                indices.append(None)
            else:
                indices.append(i)
            for check in overlap:
                res = pairing.data.get(check, {}).value == v
                if not res:
                    return
        for empty in to_fill:
            entry.data[empty] = DataItem(
                getattr(DataTypes, empty),
                [self.pairs[index].data[empty].value[0]
                 if index is not None else None for index in indices]
            )

    def _find_pairings(self, pairings: dict[Union[Static, int], Any]) -> list[DataEntry]:
        if all(isinstance(key, (Static, int)) and isinstance(val, Static)
                for key, val in pairings.items()):
            data_items = []
            for key, val in pairings.items():
                data_items += key.data + val.data if isinstance(key, Static) else val.data
            return [DataEntry(data_items)]
        pairs = []
        for val in pairings.values():
            if not isinstance(val, Static):
                pairs += self._find_pairings(val)
        return pairs

    def find_pairings(
        self,
        path: Union[str, list[str]],
        dataset: Any,
        pbar: Optional[tqdm],
        curr_path: Optional[list[str]] = None,
        in_file: bool = True,
        depth: int = 0
    ) -> None:
        '''
        Similar to other processes' `expand` function. Finds the pairing values and stores
        the data internally.
        
         - `dataset` (`Any`): the dataset data, which should follow the syntax of `DynamicData`
            data.
         - `in_file` (`bool`): distinguisher to check usage of either `expand_generics`
            or `expand_file_generics`.
        '''
        from .._main._engine import expand_generics, expand_file_generics
        if depth >= config['MAX_PBAR_DEPTH']:
            pbar = None
        if pbar:
            if curr_path is None:
                curr_path = path
            pbar.set_description(f'Expanding generics: {"/".join(curr_path)}')
        if in_file:
            expanded, _ = expand_generics(
                path,
                dataset,
                self.form,
                pbar,
                depth = depth
            )
        else:
            expanded, _ = expand_file_generics(
                path,
                curr_path,
                dataset,
                self.form,
                pbar,
                depth = depth
            )
        pairs_try = self._find_pairings(expanded)
        pairs: list[DataEntry] = []
        for pair in pairs_try:
            if self.paired.issubset({item.delimiter for item in pair.data.values()}):
                pairs.append(DataEntry([pair.data[k.desc] for k in self.paired]))
        self.pairs = pairs
