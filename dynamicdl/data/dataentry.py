from typing import Union, Iterable
from typing_extensions import Self

from .._utils import union, Warnings
from .tokens import UniqueToken, WildcardToken, RedundantToken
from .dataitem import DataItem

class DataEntry:
    '''
    Contains all items required for an entry in the dataset, a collection of DataItem objects. Most
    use is handled by internal merging processes, and is not to be instantiated by users.
    
    :param items: A (list of) data items which are to be batched together
    :type items: list[DataItem] | DataItem
    '''

    _valid_sets = [
        {'IMAGE_SET_ID', 'IMAGE_SET_NAME'},
        {'XMIN', 'XMAX', 'YMIN', 'YMAX', 'XMID', 'YMID', 'X1', 'X2', 'Y1', 'Y2', 'WIDTH', 'HEIGHT',
         'BBOX_CLASS_ID', 'BBOX_CLASS_NAME'},
        {'POLYGON', 'SEG_CLASS_ID', 'SEG_CLASS_NAME'},
        {'X', 'Y'}
    ]

    def __init__(self, items: Union[list[DataItem], DataItem]) -> None:
        items: list[DataItem] = union(items)
        self.data: dict[str, DataItem] = {item.delimiter.desc: item for item in items}
        self._update_unique()

    def _update_unique(self) -> bool:
        self.unique = any(isinstance(item.delimiter.token_type, UniqueToken)
            for item in self.data.values())

    @classmethod
    def merge(cls, first: Self, second: Self) -> Self:
        '''
        Merge two data entries together, storing it in a new instance. For inplace operations see
        `merge_inplace`.
        
        :param first: The first data entry to merge.
        :type first: DataEntry
        :param second: The second data entry to merge.
        :type second: DataEntry
        '''
        merged = cls(list(first.data.values()))
        redundant_overlap = set()
        for desc, item in second.data.items():
            if isinstance(item.delimiter.token_type, WildcardToken):
                continue
            if isinstance(item.delimiter.token_type, RedundantToken):
                if desc in merged.data and merged.data[desc] != second.data[desc]:
                    redundant_overlap.add(desc)
                continue
            if desc in merged.data and merged.data[desc] != second.data[desc]:
                Warnings.error('merge_conflict', first=first, second=second)
        allocated = False
        for group in DataEntry._valid_sets:
            if redundant_overlap.issubset(group):
                redundant_overlap = group
                allocated = True
                break
        if not allocated:
            Warnings.error(
                'merge_redundant_conflict',
                overlap=redundant_overlap,
                first=first,
                second=second
            )
        for desc in redundant_overlap:
            if desc in merged.data and desc in second.data:
                merged.data[desc].add(second.data[desc])
        for desc, item in second.data.items():
            if desc not in merged.data:
                merged.data[desc] = item
                continue
        merged._update_unique()
        return merged

    def merge_inplace(self, other: Self) -> None:
        '''
        Merge two data entries together, storing it in this instance.
        
        :param other: The other data entry to merge into this instance.
        :type other: DataEntry
        '''
        redundant_overlap = set()
        for desc, item in other.data.items():
            if isinstance(item.delimiter.token_type, WildcardToken):
                continue
            if isinstance(item.delimiter.token_type, RedundantToken):
                if desc in self.data and self.data[desc] != other.data[desc]:
                    redundant_overlap.add(desc)
                continue
            if desc in self.data and self.data[desc] != other.data[desc]:
                Warnings.error('merge_conflict', first=self, second=other)
        allocated = False
        for group in DataEntry._valid_sets:
            if redundant_overlap.issubset(group):
                redundant_overlap = group
                allocated = True
                break
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
                continue
        self._update_unique()

    def apply_tokens(self, items: Union[list[DataItem], DataItem]) -> None:
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
            if isinstance(item.delimiter.token_type, UniqueToken):
                if item.delimiter.desc in self.data and self.data[item.delimiter.desc] != item:
                    Warnings.error(
                        'merge_unique_conflict',
                        parent=self.data[item.delimiter.desc],
                        data=item
                    )
        # merge
        for item in items:
            if item.delimiter.desc not in self.data:
                if not isinstance(item.delimiter.token_type, RedundantToken):
                    self.data[item.delimiter.desc] = item
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