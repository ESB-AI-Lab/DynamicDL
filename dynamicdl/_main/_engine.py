import os
import heapq
from typing import Any, Optional, Union
from tqdm import tqdm

from .._utils import load_config
from .._warnings import Warnings, MergeError
from ..data.tokens import UniqueToken, WildcardToken
from ..data.datatype import DataType
from ..data.datatypes import DataTypes
from ..data.dataitem import DataItem
from ..data.dataentry import DataEntry
from ..parsing.static import Static
from ..parsing.generic import Generic, Folder, File
from ..parsing.namespace import Namespace
from ..parsing.genericlist import GenericList
from ..parsing.segmentationobject import SegmentationObject
from ..parsing.pairing import Pairing
from ..parsing.ambiguouslist import AmbiguousList
from ..parsing.impliedlist import ImpliedList
from ..processing.images import ImageEntry, SegmentationImage
from ..processing.datafile import DataFile
from ._utils import unique, key_has_data

config = load_config()
unique_identifiers: list[DataType] = [var for var in vars(DataTypes).values() if
    isinstance(var, DataType) and isinstance(var.token_type, UniqueToken)]
list_types = (GenericList, AmbiguousList, ImpliedList, SegmentationObject)

def expand_generics(
    path: list[str],
    dataset: Any,
    root: Any,
    xml: bool = False
) -> Union[dict, Static]:
    '''
    Expand all generics and replace with statics, inplace.
     - `dataset` (`Any`): the dataset true values, in nested blocks containing values
     - `root` (`Any`): the format of the dataset, in accordance with valid DynamicData syntax
    '''
    if isinstance(root, list):
        root = AmbiguousList(root) if xml else GenericList(root)
    if isinstance(root, list_types):
        return root.expand(
            path,
            dataset
        )
    if isinstance(root, DataType):
        root = Generic('{}', root)
    if isinstance(root, (Generic, Namespace)):
        success, tokens = root.match(str(dataset))
        if not success:
            Warnings.error('fail_generic_match', value=dataset, generic=root, path=".".join(path))
        return Static(str(dataset), tokens), []
    expanded_root: dict[Static, Any] = {}
    generics: list[Generic] = []
    names: set[Static] = set()
    pairings: list[Pairing] = []
    for i, key in enumerate(root):
        # priority queue push to prioritize generics with the most wildcards for disambiguation
        if isinstance(key, DataType):
            heapq.heappush(generics, (0, i, key))
            continue
        if isinstance(key, Generic):
            heapq.heappush(generics, (-len(key.data), i, key))
            continue
        if isinstance(key, Namespace):
            heapq.heappush(generics, (1, i, key))
            continue
        val = root[key]

        # convert str to Static
        if isinstance(key, (str, int)):
            key = Static(key)

        # add Static directly to expanded root
        if key.name in dataset:
            names.add(key.name)
            expanded_root[key] = val
            continue
        Warnings.error('static_missing', value=key, path=".".join(path))

     # expand Generics
    while len(generics) != 0:
        _, _, generic = heapq.heappop(generics)
        generic: Generic
        for name in dataset:
            # basic checks
            if name in names:
                continue

            # attempt to match name to generic
            if isinstance(generic, DataType):
                status, items = Generic('{}', generic).match(name)
            else:
                status, items = generic.match(name)

            if not status:
                continue
            names.add(name)
            expanded_root[Static(name, items)] = root[generic]

    to_pop = []

    for key, value in expanded_root.items():
        if isinstance(value, list):
            value = GenericList(value)
        if isinstance(value, (dict, Generic, Namespace)):
            uniques, pairing = expand_generics(
                path + [key.name if isinstance(key, Static) else str(key)],
                dataset[key.name],
                value
            )
            expanded_root[key] = uniques
            pairings += pairing
        elif isinstance(value, list_types):
            uniques, pairing = value.expand(
                path + [key.name if isinstance(key, Static) else str(key)],
                dataset[key.name]
            )
            expanded_root[key] = uniques
            pairings += pairing
        elif isinstance(value, DataType):
            expanded_root[key] = Static(dataset[key.name], DataItem(value, dataset[key.name]))
        elif isinstance(value, Pairing):
            value.find_pairings(
                path + [key.name if isinstance(key, Static) else str(key)],
                dataset[key.name],
                in_file = True
            )
            pairings.append(value)
            to_pop.append(key)
        else:
            Warnings.error('inappropriate_type', value=value)
    for item in to_pop:
        expanded_root.pop(item)
    return expanded_root, pairings

def _get_files(path: str) -> dict[str, Union[str, dict]]:
    '''Step one of the processing. Expand the dataset to fit all the files.'''
    files: dict[str, Union[str, dict]] = {}
    for file in os.listdir(path):
        if file.startswith('.'):
            continue
        if os.path.isdir(os.path.join(path, file)):
            files[file] = _get_files(os.path.join(path, file))
        else:
            files[file] = "File"
    return files

def expand_file_generics(
    path: str,
    curr_path: list[str],
    dataset: dict[str, Any],
    root: dict[Union[str, Static, Generic, DataType], Any],
    pbar: Optional[tqdm],
    depth: int = 0
) -> dict:
    '''
    Variant of the expand_generics function above. Also contains path tracking.'
    
     - `path` (`str`): the absolute filepath up to the root/dataset provided
     - `dataset` (`Any`): the dataset true values, in nested blocks containing values
     - `root` (`Any`): the format of the dataset, in accordance with valid DynamicData syntax
    '''
    expanded_root: dict[Static, Any] = {}
    generics: list[Generic] = []
    names: set[Static] = set()
    pairings: list[Pairing] = []
    if depth >= config['MAX_PBAR_DEPTH']:
        pbar = None
    if pbar:
        pbar.set_description(f'Expanding generics: {"/".join(curr_path)}')
    # move Statics to expanded root, move Generics to priority queue for expansion
    for i, key in enumerate(root):
        # priority queue push to prioritize generics with the most wildcards for disambiguation
        if isinstance(key, DataType):
            heapq.heappush(generics, (0, i, key))
            continue
        if isinstance(key, Generic):
            heapq.heappush(generics, (-len(key.data), i, key))
            continue
        if isinstance(key, Namespace):
            heapq.heappush(generics, (1, i, key))
            continue
        val = root[key]

        # convert str to Static
        if isinstance(key, str):
            key = Static(key)

        # add Static directly to expanded root
        if key.name in dataset:
            names.add(key.name)
            expanded_root[key] = val
            continue
        Warnings.error('static_missing', value=key, path=path)

    # expand Generics
    while len(generics) != 0:
        _, _, generic = heapq.heappop(generics)
        generic: Generic
        for name in dataset:
            # basic checks
            if name in names:
                continue
            if isinstance(generic, Folder) and dataset[name] == "File":
                continue
            if isinstance(generic, File) and dataset[name] != "File":
                continue

            # attempt to match name to generic
            if isinstance(generic, DataType):
                status, items = Generic('{}', generic).match(name)
            else:
                status, items = generic.match(name)

            if not status:
                continue
            names.add(name)
            expanded_root[Static(name, items)] = root[generic]
    to_pop = []
    # all items are statics, now process values
    for key, value in expanded_root.items():
        if isinstance(value, dict):
            next_path: str = os.path.join(path, key.name)
            uniques, pairing = expand_file_generics(
                next_path,
                curr_path + [key.name],
                dataset[key.name],
                value,
                pbar,
                depth = depth + 1
            )
            expanded_root[key] = uniques
            pairings += pairing
        elif isinstance(value, DataFile):
            uniques, pairing = value.parse(
                os.path.join(path, key.name),
                curr_path + [key.name]
            )
            expanded_root[key] = uniques
            pairings += pairing
        elif isinstance(value, ImageEntry):
            expanded_root[key] = Static(
                'ImageEntry',
                DataItem(DataTypes.ABSOLUTE_FILE, os.path.join(path, key.name))
            )
        elif isinstance(value, SegmentationImage):
            expanded_root[key] = Static(
                'Segmentation Image',
                DataItem(DataTypes.ABSOLUTE_FILE_SEG, os.path.join(path, key.name))
            )
        elif isinstance(value, Pairing):
            to_pop.append(key)
            value.find_pairings(
                os.path.join(path, key.name),
                dataset[key.name],
                pbar = pbar,
                in_file = False,
                curr_path = curr_path + [key.name],
                depth = depth + 1)
        else:
            Warnings.error('inappropriate_type', value=value)
    for item in to_pop:
        expanded_root.pop(item)
    return expanded_root, pairings

def _add_to_hashmap(
    hmap: dict[str, dict[str, DataEntry]],
    entry: DataEntry
) -> tuple[DataEntry, tuple[str, str]]:
    for item in unique(entry):
        name = entry.data[item].value
        old_entry = hmap[item].get(name, None)
        if old_entry is not None:
            old_entry.merge_inplace(entry)
            for desc in unique(old_entry):
                name = old_entry.data[desc].value
                hmap[desc][name] = old_entry
            return old_entry, (desc, name)
        hmap[item][name] = entry
    return entry, (item, name)

def _merge(
    dataset: Union[dict[Union[Static, int], Any], Static],
    path: list[str],
    data: list[DataItem],
    hmap: dict[str, dict[str, DataEntry]],
    pbar: tqdm = None,
    depth: int = 0
) -> Union[DataEntry | dict[DataEntry, tuple[str, str]]]:
    if isinstance(dataset, Static):
        entry = DataEntry(dataset.data)
        return entry
    if len(dataset) == 0:
        return DataEntry([])

    if depth >= config['MAX_PBAR_DEPTH'] or any('.' in token for token in path):
        pbar = None
    if pbar:
        pbar.set_description(f'Merging | {"/".join(path)}')

    uniques: list[DataEntry] = []
    lists: dict[DataEntry, tuple[str, str]] = {}
    others: DataEntry = DataEntry([])
    for key, val in dataset.items():
        res = _merge(
            val,
            path + [key.name if isinstance(key, Static) else str(key)],
            data + key.data if isinstance(key, Static) else [],
            hmap,
            pbar,
            depth = depth + 1
        )
        if isinstance(res, dict):
            lists.update({hmap[hloc][name]: (hloc, name) for hloc, name in res.values()})
            continue
        if key_has_data(key):
            res.apply_tokens(key.data)
        if unique(res):
            uniques.append(res)
        else:
            others.apply_tokens(res.data.values())
    if lists:
        # if we have lists and also have uniques then uniques cannot merge anymore, so add to hmap
        for item in uniques:
            entry, key = _add_to_hashmap(hmap, item)
            lists[entry] = key
        for item in lists:
            item.apply_tokens(others.data.values())
        return lists

    if not uniques:
        return others

    entry = DataEntry([])
    try:
        for item in uniques:
            entry.apply_tokens(item.data.values())
        entry.apply_tokens(others.data.values())
        return entry
    except MergeError:
        for item in uniques:
            item.apply_tokens(others.data.values())
            item.apply_tokens(data)
            entry, key = _add_to_hashmap(hmap, item)
            lists[entry] = key
        return lists

def populate_data(root_dir: str, form: dict, verbose: bool = False) -> list[DataEntry]:
    '''
    Parent process for parsing algorithm.
    
     - `root` (`str`): the file path of the root of the data
     - `form` (`dict`): the form of the data, in accordance with DynamicData syntax.
    '''
    with tqdm(
        total=4,
        desc="Getting files",
        unit="step",
        bar_format="{desc:<60.60}{percentage:3.0f}%|{bar:10}{r_bar}"
    ) as pbar:
        dataset = _get_files(root_dir)
        pbar.update(1)
        pbar.set_description('Expanding generics')
        data, pairings = expand_file_generics(
            root_dir,
            [],
            dataset,
            form,
            pbar if verbose else None
        )
        if verbose:
            from .._utils import get_str
            print("Expanded data to:")
            print(get_str(data))
        pbar.update(1)
        pbar.set_description('Merging data')
        hmap: dict[str, dict[str, DataEntry]] = {id.desc:{} for id in unique_identifiers}
        data = _merge(data, [], [], hmap, pbar, depth=1)
        if isinstance(data, DataEntry):
            Warnings.error('merged_all')
        data = [hmap[hloc][name] for hloc, name in data.values()]
        pbar.update(1)
        pbar.set_description('Applying pairing entries')
        for pairing in pairings:
            for entry in data:
                pairing.update_pairing(entry)
        pbar.set_description('Removing all generic values')
        for entry in data:
            entry.data = {k: v for k, v in entry.data.items()
                if not isinstance(v.delimiter.token_type, WildcardToken)}
        pbar.update(1)
        pbar.set_description('Done!')
    return data
