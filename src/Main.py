import os
import heapq
from typing import Any

from .Names import Static, Generic
from .Processing import TXTFile, JSONFile, Image

def _get_files(path: str) -> dict:
    files = {}
    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            files[file] = _get_files(os.path.join(path, file))
        else:
            files[file] = "File"
    return files

def _expand_generics(path: str, dataset: dict[str, Any],
                     root: dict[str | Static | Generic, Any]) -> dict:
    '''
    Expand all generics and set to statics.
    '''
    expanded_root: dict[Static, Any] = {}
    generics: list[Generic] = []
    names: set[Static] = set()
    for key in root:
        if isinstance(key, Generic):
            # priority queue push to prioritize generics with the most wildcards for disambiguation
            heapq.heappush(generics, (-len(key.data), key))
            continue
        if isinstance(key, str):
            if key.name in dataset:
                names.add(key)
                expanded_root[Static(key)] = root[key]
            else:
                raise ValueError(f'Static value {key} not found in dataset')
        if key.name in dataset:
            names.add(key.name)
            expanded_root[key] = root[key]
        else:
            raise ValueError(f'Static value {key} not found in dataset')

    while len(generics) != 0:
        _, generic = heapq.heappop(generics)
        generic: Generic
        for name in dataset:
            if name in names: continue
            status, items = generic.match(name)
            if not status: continue
            new_name: str = generic.substitute(items)
            names.add(new_name)
            expanded_root[Static(new_name, items)] = root[generic]

    for key, value in expanded_root.items():
        if isinstance(value, dict):
            expanded_root[key] = _expand_generics(os.path.join(path, key.name),
                                                  dataset[key.name], expanded_root[key])
        elif isinstance(value, (TXTFile, JSONFile)):
            expanded_root[key] = value.parse(os.path.join(path, key.name))
        elif isinstance(value, Image):
            expanded_root[key] = Image()
    return expanded_root

class Dataset:
    def __init__(self, root: str, form: dict[Static | Generic, Any]):
        self.root = root
        self.form = form
        self.dataset = self._expand()

    def _expand(self) -> dict:
        '''
        Expand dataset into dict format
        '''
        dataset = _get_files(self.root)
        return _expand_generics(self.root, dataset, self.form)
