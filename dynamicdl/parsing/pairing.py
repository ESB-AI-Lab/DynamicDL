from typing import Any, Optional, Union
from tqdm import tqdm

from .._utils import load_config
from .._warnings import Warnings, MergeError
from ..data.tokens import RedundantToken
from ..data.datatype import DataType
from ..data.dataitem import DataItem
from ..data.dataentry import DataEntry
from .._utils import key_has_data, union
from .static import Static

config = load_config()

class Pairing:
    '''
    Pairing is a wrapper class used to specify when two or more nonunique datatypes should be
    associated together. Most commonly used to pair ID and name together. 

    :param form: Whatever follows the DynamicData specified form as required. Pairing is a
        wrapper class so let it behave as it should.
    :type form: Any
    :param paired: Items which should be associated together.
    :type paired: DataType
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
        Update a data entry with pairing values, and does nothing if the pairing does not apply.
        
        :param entry: The entry to apply this pairing
        :type entry: DataEntry
        '''
        if not self.pairs:
            Warnings.error('empty_pairing', path=self.pairing_path)
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
                if v == pairing.data[overlap[0]].value[0]:
                    index = i
                    break
            if index == -1:
                indices.append(None)
            else:
                indices.append(i)
            for check in overlap:
                item = pairing.data.get(check, None)
                if item is None:
                    continue
                if not item.value[0] == v:
                    return
        for empty in to_fill:
            entry.data[empty] = DataItem(
                DataType.types[empty],
                [self.pairs[index].data[empty].value[0]
                 if index is not None else None for index in indices]
            )

    def _merge(
        self,
        dataset: Union[dict[Union[Static, int], Any], Static],
        data: list[DataItem]
    ) -> Union[DataEntry | dict[DataEntry, tuple[str, str]]]:
        if isinstance(dataset, Static):
            entry = DataEntry(dataset.data)
            return entry
        if len(dataset) == 0:
            return DataEntry([])

        uniques: list[DataEntry] = []
        lists: dict[DataEntry, tuple[str, str]] = {}
        for key, val in dataset.items():
            res = self._merge(
                val,
                data + key.data if isinstance(key, Static) else []
            )
            if isinstance(res, dict):
                continue
            if key_has_data(key):
                res.apply_tokens(key.data)
            uniques.append(res)
        if lists:
            for item in uniques:
                lists[entry] = key
            return lists

        entry = DataEntry([])
        try:
            for item in uniques:
                entry.apply_tokens(item.data.values())
            return entry
        except MergeError:
            for item in uniques:
                item.apply_tokens(data)
                lists[entry] = key
            return lists

    def find_pairings(
        self,
        path: Union[str, list[str]],
        dataset: Any,
        pbar: Optional[tqdm] = None,
        curr_path: Optional[list[str]] = None,
        in_file: bool = True,
        depth: int = 0
    ) -> None:
        '''
        Similar to other processes' `expand` function. Finds the pairing values and stores
        the data internally.
        
        :param dataset: The dataset data, which should follow the syntax of `DynamicData` data.
        :type dataset: Any
        :param in_file: Distinguisher to check usage of either `expand_generics`
            or `expand_file_generics`.
        :type in_file: bool
        '''
        from .._main._engine import expand_generics, expand_file_generics
        if in_file:
            expanded, _ = expand_generics(
                path,
                dataset,
                self.form
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
        pairs_try: Union[list[DataEntry], DataEntry] = self._merge(expanded, [])
        if self.redundant:
            entries = []
            for items in zip(*[list(map(lambda x: (v.delimiter, x), v.value))
                               for v in pairs_try.data.values()]):
                dataitems = []
                for item in items:
                    dataitems.append(DataItem(*item))
                entries.append(DataEntry(dataitems))
        else:
            entries = union(pairs_try)
        pairs: list[DataEntry] = []
        for pair in entries:
            if self.paired.issubset({item.delimiter for item in pair.data.values()}):
                pairs.append(DataEntry([pair.data[k.desc] for k in self.paired]))
        self.pairs = pairs
        self.pairing_path = ".".join(path) if in_file else path
