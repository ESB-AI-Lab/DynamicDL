'''
Main module for processing datasets.
'''
import os
import heapq
import time
from typing import Any
import json
from pandas import DataFrame
from pandas.core.series import Series
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, IntTensor, FloatTensor
import torch

from .DataItems import DataEntry, DataItem, DataTypes, DataType, UniqueToken, Static, Generic, Image
from .Processing import TXTFile, JSONFile

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
            if key in dataset:
                names.add(key)
                expanded_root[Static(key)] = root[key]
            else: raise ValueError(f'Static value {key} not found in dataset')
            continue
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
            expanded_root[key] = Static('Image', DataItem(DataTypes.ABSOLUTE_FILE,
                                                           os.path.join(path, key.name)))
    return expanded_root

def _make_uniform(data: dict[Static, Any] | list[Any] | Static) -> dict:
    if isinstance(data, dict):
        items = {}
        for key, val in data.items():
            items[key] = _make_uniform(val)
        return items
    if isinstance(data, list):
        return {i:_make_uniform(item) for i, item in enumerate(data)}
    return data

def _split(data: dict[Static | int, Any] | Static) -> tuple[dict, dict]:
    if isinstance(data, Static):
        return (None, data) if DataEntry(data.data).unique else (data, None)
    count = 0
    pairings = {}
    uniques = {}
    for key, val in data.items():
        if isinstance(key, Static) and DataEntry(key.data).unique:
            count += 1
            uniques[key] = val
            continue
        if isinstance(val, Static) and DataEntry(val.data).unique:
            count += 1
            uniques[key] = val
            continue
        pairs, unique_vals = _split(val)
        if pairs: pairings[key] = pairs
        if unique_vals: uniques[key] = unique_vals
    if count == 1:
        return (None, data)
    return pairings, uniques

def _find_pairings(pairings: dict[Static | int, Any], curr_data: list[DataItem]) -> list[DataEntry]:
    if all([isinstance(key, (Static, int)) and isinstance(val, Static)
            for key, val in pairings.items()]):
        data_items = []
        for key, val in pairings.items():
            data_items += key.data + val.data if isinstance(key, Static) else val.data
        return [DataEntry(data_items + curr_data)]
    pairs = []
    structs = []
    for key, val in pairings.items():
        if isinstance(key, Static): curr_data += key.data
        if isinstance(val, Static): curr_data += val.data
        else: structs.append(val)

    for struct in structs:
        pairs += _find_pairings(struct, curr_data)
    return pairs

def _add_to_hashmap(hashmaps: dict[str, dict[str, DataEntry]], entry: DataEntry,
                    unique_identifiers: list[DataType]) -> None:
    '''
    Helper method for _merge_lists()
    '''
    for id_try in unique_identifiers:
        value = entry.data.get(id_try.desc)
        if not value: continue
        if value.value in hashmaps[id_try.desc]:
            result = hashmaps[id_try.desc][value.value].merge_inplace(entry)
            if not result: raise ValueError(f'Found conflicting information when merging \
                {hashmaps[id_try.desc][value.value]} and {entry}')
            for id_update in unique_identifiers:
                value_update = entry.data.get(id_update.desc)
                if id_update == id_try or not value_update: continue
                hashmaps[id_update.desc][value_update.value] = hashmaps[id_try.desc][value.value]
            break
        hashmaps[id_try.desc][value.value] = entry

def _merge_lists(lists: list[list[DataEntry]]) -> list[DataEntry]:
    '''
    Merge two DataEntry lists.
    '''
    if len(lists) == 0: return []
    if len(lists) == 1: return lists[0]
    # needs to be changed if allow custom definition of datatypes
    unique_identifiers: list[DataType] = [var for var in vars(DataTypes).values() if
                                            isinstance(var, DataType) and
                                            isinstance(var.token_type, UniqueToken)]
    hashmaps: dict[str, dict[str, DataEntry]] = {id.desc:{} for id in unique_identifiers}
    for next_list in lists:
        for entry in next_list:
            _add_to_hashmap(hashmaps, entry, unique_identifiers)

    data = set()
    for identifier in unique_identifiers:
        data.update(hashmaps[identifier.desc].values())
    return list(data)

def _merge(data: dict[Static | int, Any] | Static, pairings: list[DataEntry]) -> \
        DataEntry | list[DataEntry]:
    if isinstance(data, Static):
        entry = DataEntry(data.data) # apply pairings here if possible
        return entry
    if len(data) == 0: return []
    recursive = []
    for key, val in data.items():
        result = _merge(val, pairings)
        if isinstance(result, DataEntry):
            recursive.append(DataEntry.merge(DataEntry(key.data), result)
                             if isinstance(key, Static) else result)
            continue
        if isinstance(key, Static):
            for item in result: item.apply_tokens(key.data)
        recursive.append(result)
    lists = [item for item in recursive if isinstance(item, list)]
    tokens = [item for item in recursive if not isinstance(item, list)]
    if lists:
        result = _merge_lists(lists)
        if tokens:
            for token in tokens:
                for item in result: item.apply_tokens(token)
        return result
    entries = []
    entry = recursive[0]
    for index, item in enumerate(recursive[1:], 1):
        res = DataEntry.merge(entry, item, overlap=False)
        if res:
            entry = res
            continue
        entries.append(entry)
        entry = recursive[index]
    entries.append(entry)
    return entries if len(entries) > 1 else entries[0]

def _get_str(data):
    if isinstance(data, dict):
        return {str(key): _get_str(val) for key, val in data.items()}
    if isinstance(data, list):
        return [_get_str(val) for val in data]
    return str(data)

def get_str(data):
    '''Return pretty print string.'''
    return json.dumps(_get_str(data), indent=4).replace('"', '')

def _get_class_labels(item: Series) -> Tensor:
    return int(item['CLASS_ID'])

def _get_bbox_labels(item: dict) -> dict[str, Tensor]:
    # execute checks
    class_ids = item['BBOX_CLASS_ID']
    xmins = item['XMIN']
    ymins = item['YMIN']
    xmaxs = item['XMAX']
    ymaxs = item['YMAX']
    bbox_tensors = [FloatTensor(list(zip(xmins, ymins, xmaxs, ymaxs)))]
    return {'boxes': torch.stack(bbox_tensors),
            'labels': IntTensor(class_ids)}

def _collate(batch):
    return tuple(zip(*batch))

def _purge_invalid_bbox(df: DataFrame) -> DataFrame:
    '''Purge all entries without valid bbox entries'''
    valid = [True] * len(df)
    for i, row in df[['BBOX_CLASS_ID', 'XMIN', 'XMAX', 'YMIN', 'YMAX']].iterrows():
        if not len(row['BBOX_CLASS_ID']) == len(row['XMIN']) == len(row['XMAX']) == \
            len(row['YMIN']) == len(row['YMAX']):
            valid[i] = False
    return df[valid]

class CVData:
    '''
    Main dataset class utils.
    '''

    _classification_cols = {'ABSOLUTE_FILE', 'IMAGE_ID', 'CLASS_ID'}
    _detection_cols = {'ABSOLUTE_FILE', 'IMAGE_ID', 'BBOX_CLASS_ID', 'XMIN', 'XMAX', 'YMIN', 'YMAX'}

    def __init__(self, root: str, form: dict[Static | Generic, Any], transform=None,
                 target_transform=None, remove_invalid=True):
        self.root = root
        self.form = form
        self.transform = transform
        self.target_transform = target_transform
        dataset = _get_files(self.root)
        data = _make_uniform(_expand_generics(self.root, dataset, self.form))
        pairings, uniques = _split(data)
        pairings = _find_pairings(pairings, [])
        self.data: list[DataEntry] = _merge(uniques, pairings)
        self.dataframe: DataFrame = DataFrame([{key: val.value if isinstance(val, DataItem)
                                                else [x.value for x in val]
                                                for key, val in data.data.items()}
                                               for data in self.data])
        self.image_set_to_idx = {}
        self.image_sets = set()
        self.remove_invalid = remove_invalid
        self.available_modes = []
        if CVData._classification_cols.issubset(self.dataframe.columns):
            self.available_modes.append('classification')
        if CVData._detection_cols.issubset(self.dataframe.columns):
            self.available_modes.append('detection')
        # add segmentation mode
        self.cleaned = False

    def cleanup(self) -> None:
        '''
        Run cleanup and sanity checks on all data.
        '''
        print('[CVData] Cleaning up data...')
        self._cleanup_image_sets()
        self.cleaned = True
        print('[CVData] Done!')

    def _cleanup_image_sets(self) -> None:
        if 'IMAGE_SET_ID' not in self.dataframe:
            if 'IMAGE_SET' not in self.dataframe:
                print('[CVData] No image set found. Assigning to default image set')
                self.dataframe['IMAGE_SET'] = [['default']] * len(self.dataframe)
                self.dataframe['IMAGE_SET_ID'] = [[0]] * len(self.dataframe)
                self.image_set_to_idx = {'default': 0}
                self.image_sets = {'default'}
            else:
                print('[CVData] Converting IMAGE_SET names to IMAGE_SET_ID')
                self.dataframe.loc[self.dataframe['IMAGE_SET'].isna(), 'IMAGE_SET'] = \
                    self.dataframe.loc[self.dataframe['IMAGE_SET'].isna(), 'IMAGE_SET'].apply(
                        lambda x: ['default'])
                for v in self.dataframe['IMAGE_SET']:
                    self.image_sets.update(v)
                self.image_set_to_idx = {v: i for i, v in enumerate(self.image_sets)}
                self.dataframe['IMAGE_SET_ID'] = self.dataframe['IMAGE_SET'].apply(
                    lambda x: list(map(lambda y: self.image_set_to_idx[y], x)))
        elif 'IMAGE_SET' in self.dataframe:
            for ids, vals in self.dataframe[['IMAGE_SET_ID', 'IMAGE_SET']]:
                if len(ids) != len(vals): raise ValueError('Invalid image sets lengths in row')
                for i, v in zip(ids, vals):
                    if v in self.image_set_to_idx and self.image_set_to_idx[v] != i:
                        raise ValueError(f'Invalid image_set id {i} assigned to name {v}')
                    else: self.image_set_to_idx[v] = i
                self.image_sets = set(self.image_set_to_idx.keys())
        else:
            for ids in self.dataframe['IMAGE_SET_ID']:
                self.image_sets.update(ids)

    def get_dataset(self, mode: str, image_set: str) -> Dataset:
        '''
        Retrieve the dataset.
        '''
        if not self.cleaned: self.cleanup()
        assert mode.lower().strip() in self.available_modes, 'Desired mode not available.'
        dataframe = self.dataframe[[image_set in item for item in self.dataframe['IMAGE_SET']]]
        if len(dataframe) == 0: raise ValueError(f'Image set {image_set} not available.')
        match mode:
            case 'classification':
                dataframe = dataframe.drop(['BBOX_CLASS_ID', 'BBOX_CLASS_NAME', 'XMIN', 'XMAX',
                                            'YMIN', 'YMAX', 'WIDTH', 'HEIGHT'], 
                                           inplace=False, errors='ignore')
                if self.remove_invalid:
                    print(f'Removed {len(dataframe[dataframe.isna().any(axis=1)])} NaN entries.')
                    dataframe = dataframe.dropna()
            case 'detection':
                dataframe = dataframe.drop(['CLASS_ID', 'CLASS_NAME'],
                                           inplace=False, errors='ignore')
                if self.remove_invalid:
                    print(f'Removed {len(dataframe[dataframe.isna().any(axis=1)])} NaN entries.')
                    dataframe = dataframe.dropna()
                dataframe = _purge_invalid_bbox(dataframe)
        if len(dataframe) == 0: raise ValueError('[CVData] After cleanup, this dataset is empty.')
        return CVDataset(dataframe, mode)

    def get_dataloader(self, mode: str, image_set: str, batch_size: int = 4, shuffle: bool = True,
                       num_workers: int = 1) -> DataLoader:
        '''
        Retrieve the dataloader for this dataset.
        '''
        return DataLoader(self.get_dataset(mode, image_set), batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers, collate_fn=_collate)

class CVDataset(Dataset):
    '''
    Dataset implementation for CVData environment.
    '''
    def __init__(self, df: DataFrame, mode: str):
        self.dataframe = df
        self.data = self.dataframe.to_dict('records')
        self.mode = mode

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        start = time.time()
        item: dict = self.data[idx]
        image: Tensor = read_image(item.get('ABSOLUTE_FILE'))
        label: dict[str, Tensor]
        match self.mode:
            case 'classification':
                label = _get_class_labels(item)
            case 'detection':
                label = _get_bbox_labels(item)
            case 'segmentation':
                pass
        end = time.time()
        print(f'Seconds: {end - start}')
        return image, label
