'''
.. module:: DataEntry

'''

from typing import Union, Iterable
from typing_extensions import Self

from .._utils import union, config
from .._warnings import Warnings
from .tokens import UniqueToken, WildcardToken, RedundantToken
from .partialtype import PartialType
from .dataitem import DataItem

class DataEntry:
    '''
    Contains all items required for an entry in the dataset, a collection of DataItem objects. Most
    use is handled by internal merging processes, and is not to be instantiated by users.
    
    :param items: A (list of) data items which are to be batched together
    :type items: list[DataItem] | DataItem
    '''

    _valid_sets = config['VALID_ENTRY_SETS']

    def __init__(self, items: Union[list[DataItem], DataItem]) -> None:
        items: list[DataItem] = union(items)
        self.data: dict[str, DataItem] = {item.delimiter.desc: item for item in items}
        self._update_unique()

    def _update_unique(self) -> bool:
        self.unique = any(isinstance(item.delimiter.token_type, UniqueToken)
            for item in self.data.values())

    def merge_inplace(self, other: Self) -> None:
        '''
        Merge two data entries together, storing it in this instance.
        
        :param other: The other data entry to merge into this instance.
        :type other: DataEntry
        '''
        redundant_overlap: set[Union[str, PartialType]] = set()
        for desc, item in other.data.items():
            if isinstance(item.delimiter.token_type, WildcardToken):
                continue
            if isinstance(item.delimiter.token_type, RedundantToken):
                if desc in self.data and self.data[desc] != other.data[desc]:
                    if isinstance(item.delimiter, PartialType):
                        desc = item.delimiter
                    redundant_overlap.add(desc)
                continue
            if desc in self.data and self.data[desc] != other.data[desc]:
                Warnings.error('merge_conflict', first=self, second=other)

        if redundant_overlap:
            allocated = False
            for group in DataEntry._valid_sets:
                if redundant_overlap.issubset(group):
                    redundant_overlap = group
                    allocated = True
                    break
            # catch partial types; they belong to same group if parents are all same
            if not allocated and all(isinstance(dt, PartialType) for dt in redundant_overlap):
                first = redundant_overlap.pop()
                redundant_overlap.add(first)
                if all(first.parent == dt.parent for dt in redundant_overlap):
                    allocated = True
            if not allocated:
                Warnings.error(
                    'merge_redundant_conflict',
                    overlap=redundant_overlap,
                    first=self,
                    second=other
                )
            for desc in redundant_overlap:
                if desc in self.data and desc in other.data:
                    self.data[desc].add(other.data[desc])
        for desc, item in other.data.items():
            if desc not in self.data:
                self.data[desc] = item
                if isinstance(item.delimiter, PartialType):
                    self._handle_partial_types(item.delimiter)
                continue
        self._update_unique()

    def _handle_partial_types(self, datatype: PartialType) -> None:
        parent = datatype.parent
        if set(map(lambda x: x, parent.datatypes)).issubset(self.data.keys()):
            values = [self.data[dt].value for dt in parent.datatypes]
            item = DataItem(parent.to, parent.construct(values))
            # require recursive apply tokens to prevent merge conflicts
            self.apply_tokens([item])
            if parent.preserve_all:
                return
            for dt in parent.datatypes:
                self.data.pop(dt)

    def apply_tokens(self, items: Iterable[DataItem]) -> None:
        '''
        Apply new tokens to the item.
        
        :param items: Additional items to associate with this data entry.
        :type items: list[DataItem] | DataItem
        '''
        if not isinstance(items, Iterable):
            items = [items]
        items: list[DataItem] = [DataItem.copy(item) for item in items]
        # execute checks first
        for item in items:
            if isinstance(item.delimiter.token_type, RedundantToken):
                continue
            if item.delimiter.desc in self.data and self.data[item.delimiter.desc] != item:
                Warnings.error(
                    'merge_unique_conflict',
                    parent=self.data[item.delimiter.desc],
                    token=item
                )
        # merge
        for item in items:
            if item.delimiter.desc not in self.data:
                if not isinstance(item.delimiter.token_type, RedundantToken):
                    self.data[item.delimiter.desc] = item
                    if isinstance(item.delimiter, PartialType):
                        self._handle_partial_types(item.delimiter)
                    continue
                for group in DataEntry._valid_sets:
                    if item.delimiter.desc in group:
                        break
                # redundant token must fall into one of these groups so no error checking
                # if none of the groups already exist then default to 1x application otherwise
                # must match length with other items in the group
                n = 1
                matched = False
                for desc in group:
                    if desc in self.data:
                        n = len(self.data[desc].value)
                        matched = True
                        break
                assert not matched or len(item.value) == 1 or n == len(item.value), \
                    ('Assertion failed (report as a bug!) - (len(item.value) == 1);'
                     f'item: {item} | group: {group} | self: {self}')

                if not matched or n == len(item.value):
                    self.data[item.delimiter.desc] = DataItem(
                        item.delimiter,
                        item.value
                    )
                    continue
                self.data[item.delimiter.desc] = DataItem(
                    item.delimiter,
                    item.value * n
                )
            elif isinstance(item.delimiter.token_type, RedundantToken):
                self.data[item.delimiter.desc].add(item)
        self._update_unique()

    def __repr__(self) -> str:
        return ' '.join(['DataEntry:']+[str(item) for item in self.data.values()])
