import os
import heapq
from typing import Any, Optional, Union
from tqdm import tqdm

from ._utils import Warnings, check_map, load_config
from .data.tokens import UniqueToken
from .data.datatype import DataType
from .data.datatypes import DataTypes
from .data.dataitem import DataItem
from .data.dataentry import DataEntry
from .parsing.static import Static
from .parsing.generic import Generic, Folder, File
from .parsing.namespace import Namespace
from .parsing.genericlist import GenericList
from .parsing.segmentationobject import SegmentationObject
from .parsing.pairing import Pairing
from .parsing.ambiguouslist import AmbiguousList
from .processing.images import ImageEntry, SegmentationImage
from .processing.datafile import DataFile

config = load_config()

def expand_generics(
    path: list[str],
    dataset: Any,
    root: Any,
    pbar: Optional[tqdm] = None,
    depth: int = 0
) -> Union[dict, Static]:
    '''
    Expand all generics and replace with statics, inplace.
     - `dataset` (`Any`): the dataset true values, in nested blocks containing values
     - `root` (`Any`): the format of the dataset, in accordance with valid DynamicData syntax
    '''
    if isinstance(root, list):
        root = GenericList(root)
    if isinstance(root, (GenericList, SegmentationObject)):
        return root.expand(
            path,
            dataset,
            pbar,
            depth = depth
        )
    if isinstance(root, DataType):
        root = Generic('{}', root)
    if isinstance(root, (Generic, Namespace)):
        success, tokens = root.match(str(dataset))
        if not success:
            Warnings.error('fail_generic_match', dataset=dataset, root=root)
        return Static(str(dataset), tokens), []
    expanded_root: dict[Static, Any] = {}
    generics: list[Generic] = []
    names: set[Static] = set()
    pairings: list[Pairing] = []
    if depth >= config['MAX_PBAR_DEPTH']:
        pbar = None
    if pbar:
        pbar.set_description(f'Expanding generics: {"/".join(path)}')
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
        Warnings.error('static_missing', value=key)

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
                value,
                pbar,
                depth = depth + 1
            )
            expanded_root[key] = uniques
            pairings += pairing
        elif isinstance(value, (GenericList, AmbiguousList, SegmentationObject)):
            uniques, pairing = value.expand(
                path + [key.name if isinstance(key, Static) else str(key)],
                dataset[key.name],
                pbar,
                depth = depth + 1
            )
            expanded_root[key] = uniques
            pairings += pairing
        elif isinstance(value, DataType):
            expanded_root[key] = Static(dataset[key.name], DataItem(value, dataset[key.name]))
        elif isinstance(value, Pairing):
            value.find_pairings(
                path + [key.name if isinstance(key, Static) else str(key)],
                dataset[key.name],
                pbar,
                in_file = True,
                depth = depth + 1
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
        Warnings.error('static_missing', value=key)

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
                curr_path + [key.name],
                pbar,
                depth = depth + 1,
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
                pbar,
                in_file = False,
                curr_path = curr_path + [key.name],
                depth = depth + 1)
        else:
            Warnings.error('inappropriate_type', value=value)
    for item in to_pop:
        expanded_root.pop(item)
    return expanded_root, pairings

def _add_to_hashmap(hashmaps: dict[str, dict[str, DataEntry]], entry: DataEntry,
                    unique_identifiers: list[DataType]) -> None:
    for id_try in unique_identifiers:
        value = entry.data.get(id_try.desc)
        if not value:
            continue
        if value.value in hashmaps[id_try.desc]:
            hashmaps[id_try.desc][value.value].merge_inplace(entry)
            for id_update in unique_identifiers:
                value_update = entry.data.get(id_update.desc)
                if id_update == id_try or not value_update:
                    continue
                hashmaps[id_update.desc][value_update.value] = hashmaps[id_try.desc][value.value]
            break
        hashmaps[id_try.desc][value.value] = entry

def _merge_lists(lists: list[list[DataEntry]]) -> list[DataEntry]:
    if len(lists) == 0:
        return []

    # get all unique identifiers
    unique_identifiers: list[DataType] = [var for var in vars(DataTypes).values() if
        isinstance(var, DataType) and isinstance(var.token_type, UniqueToken)]

    # append to hashmaps for efficient merge
    hashmaps: dict[str, dict[str, DataEntry]] = {id.desc:{} for id in unique_identifiers}
    for next_list in lists:
        for entry in next_list:
            _add_to_hashmap(hashmaps, entry, unique_identifiers)

    # extract data from all hashmaps, same entries have same pointer so set works for unique items
    data = set()
    for identifier in unique_identifiers:
        data.update(hashmaps[identifier.desc].values())
    return list(data)

def _merge(
    data: Union[dict[Union[Static, int], Any], Static],
    path: list[str],
    depth: int = 0
) -> Union[DataEntry, list[DataEntry]]:
    # base cases
    if isinstance(data, Static):
        return DataEntry(data.data)
    if len(data) == 0:
        return []
    recursive = []

    pbar = data.items()
    if len(data) > 100:
        pbar = tqdm(data.items(), desc="/".join(path), position=depth, leave=False)
    # get result
    for key, val in pbar:
        result = _merge(
            val,
            path + [key.name if isinstance(key, Static) else str(key)],
            depth = depth + 1
        )
        # unique entry result
        if isinstance(result, DataEntry):
            if isinstance(key, Static):
                result.apply_tokens(key.data)
                # result = DataEntry.merge(DataEntry(key.data), result)
            if result.unique:
                recursive.append([result])
            else: recursive.append(result)
            continue
        # list entry result
        if isinstance(key, Static):
            for item in result:
                item.apply_tokens(key.data)
        recursive.append(result)
    lists: list[list[DataEntry]] = []
    tokens: list[DataEntry] = []
    for item in recursive:
        if isinstance(item, list):
            lists.append(item)
        else:
            tokens.append(item)

    # if outside unique loop, merge lists and apply tokens as needed
    if lists:
        result = _merge_lists(lists)
        if tokens:
            for item in result:
                for token in tokens:
                    item.apply_tokens(token.data.values())
        return result

    # if inside unique loop, either can merge all together or result has multiple entries
    if not check_map((item.unique for item in tokens), 2):
        result = DataEntry([])
        for item in tokens:
            result = DataEntry.merge(item, result)
        return result

    # if there are non unique entries then the data will naturally fall through as pairings are
    # meant to catch the nonunique data
    uniques: list[DataEntry] = []
    others: list[DataEntry] = []
    for item in tokens:
        if item.unique:
            uniques.append(item)
        else:
            others.append(item)
    for other in others:
        for entry in uniques:
            entry.apply_tokens(other.data.values())
    return uniques

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
        pbar.update(1)
        pbar.set_description('Merging data')
        data = _merge(data, [], depth=1)
        pbar.update(1)
        pbar.set_description('Applying pairing entries')
        for pairing in pairings:
            for entry in data:
                pairing.update_pairing(entry)
        pbar.update(1)
        pbar.set_description('Done!')
    return _merge_lists([data])
