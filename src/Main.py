'''
Main module for processing datasets.
'''
import os
import heapq
from typing import Any, Union
from math import isnan
import json
from numpy import asarray, int32, full_like
from pandas import DataFrame
from pandas.core.series import Series
from cv2 import imread, fillPoly, IMREAD_GRAYSCALE
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, IntTensor, FloatTensor
import torch


from .DataItems import DataEntry, DataItem, DataTypes, DataType, UniqueToken, Static, Generic, \
                       Image, SegmentationImage, Folder, File
from .Processing import TXTFile, JSONFile, Pairing

def _get_files(path: str) -> dict:
    files = {}
    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            files[file] = _get_files(os.path.join(path, file))
        else:
            files[file] = "File"
    return files

def _expand_generics(path: str, dataset: dict[str, Any],
                     root: dict[Union[str, Static, Generic], Any]) -> dict:
    '''
    Expand all generics and set to statics.
    '''
    expanded_root: dict[Static, Any] = {}
    generics: list[Generic] = []
    names: set[Static] = set()
    pairings: list[Pairing] = []
    for i, key in enumerate(root):
        if isinstance(key, DataType):
            heapq.heappush(generics, (0, i, Generic('{}', key)))
        if isinstance(key, Generic):
            # priority queue push to prioritize generics with the most wildcards for disambiguation
            heapq.heappush(generics, (-len(key.data), i, key))
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
        _, _, generic = heapq.heappop(generics)
        generic: Generic
        for name in dataset:
            if name in names: continue
            if isinstance(generic, Folder) and dataset[name] == "File": continue
            if isinstance(generic, File) and dataset[name] != "File": continue
            status, items = generic.match(name)
            if not status: continue
            new_name: str = generic.substitute(items)
            names.add(new_name)
            expanded_root[Static(new_name, items)] = root[generic]

    to_pop = []

    for key, value in expanded_root.items():
        if isinstance(value, dict):
            uniques, pairing = _expand_generics(os.path.join(path, key.name),
                                                           dataset[key.name], expanded_root[key])
            expanded_root[key] = uniques
            pairings += pairing
        elif isinstance(value, (TXTFile, JSONFile)):
            uniques, pairing = value.parse(os.path.join(path, key.name))
            expanded_root[key] = uniques
            pairings += pairing
        elif isinstance(value, Image):
            expanded_root[key] = Static('Image', DataItem(DataTypes.ABSOLUTE_FILE,
                                                           os.path.join(path, key.name)))
        elif isinstance(value, SegmentationImage):
            expanded_root[key] = Static('Segmentation Image', DataItem(DataTypes.ABSOLUTE_FILE_SEG,
                                                           os.path.join(path, key.name)))
        elif isinstance(value, Pairing):
            to_pop.append(key)
            value.find_pairings(dataset[key.name])
            
    return expanded_root, pairings

def _split(data: Union[dict[Union[Static, int], Any], Static]) -> tuple[dict, dict]:
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

def _merge(data: Union[dict[Union[Static, int], Any], Static]) -> \
        Union[DataEntry, list[DataEntry]]:
    if isinstance(data, Static):
        entry = DataEntry(data.data) # apply pairings here if possible
        return entry
    if len(data) == 0: return []
    recursive = []
    for key, val in data.items():
        result = _merge(val)
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

def _get_bbox_labels(item: Series) -> dict[str, Tensor]:
    # execute checks
    assert len(item['BOX']) == len(item['BBOX_CLASS_ID']), \
        'SEG_CLASS_ID and POLYGON len mismatch'
    class_ids = item['BBOX_CLASS_ID']
    boxes = item['BOX']
    bbox_tensors = [FloatTensor(box) for box in boxes]
    return {'boxes': torch.stack(bbox_tensors), 'labels': IntTensor(class_ids)}

def _get_seg_labels(item: Series, default=0) -> Tensor:
    if 'ABSOLUTE_FILE_SEG' in item:
        return read_image(item['ABSOLUTE_FILE_SEG'])
    assert len(item['POLYGON']) == len(item['SEG_CLASS_ID']), \
        'SEG_CLASS_ID and POLYGON len mismatch'
    mask = asarray(imread(item['ABSOLUTE_FILE'], IMREAD_GRAYSCALE), dtype=int32)
    mask = full_like(mask, default)
    for class_id, polygon in zip(item['SEG_CLASS_ID'], item['POLYGON']):
        mask = fillPoly(mask, pts=[asarray(polygon, dtype=int32)], color=class_id)
    mask = torch.from_numpy(asarray(mask))
    return mask
def _collate(batch):
    return tuple(zip(*batch))

class CVData:
    '''
    Main dataset class utils.
    '''

    _classification_cols = {'ABSOLUTE_FILE', 'IMAGE_ID', 'CLASS_ID'}
    _detection_cols = {'ABSOLUTE_FILE', 'IMAGE_ID', 'BBOX_CLASS_ID', 'BOX'}
    _segmentation_cols = {'ABSOLUTE_FILE', 'IMAGE_ID'}

    def __init__(self, root: str, form: dict[Union[Static, Generic], Any], transform=None,
                 target_transform=None, remove_invalid=True):
        self.root = root
        self.form = form
        self.transform = transform
        self.target_transform = target_transform
        dataset = _get_files(self.root)
        data, pairings = _expand_generics(self.root, dataset, self.form)
        self.data: list[DataEntry] = _merge(data)
        for pairing in pairings:
            for entry in self.data:
                pairing.update_pairing(entry)
        # self.data = _apply_pairings(uniques, pairings)
        self.dataframe: DataFrame = DataFrame([{key: val.value if isinstance(val, DataItem)
                                                else [x.value for x in val]
                                                for key, val in data.data.items()}
                                               for data in self.data])
        self.image_set_to_idx = {}
        self.image_sets = []
        self.remove_invalid = remove_invalid
        self.available_modes = []
        self.cleaned = False

    def _assign_ids(self, name: str, default=False, redundant=False, assign=False) -> \
            tuple[dict, set]:
        if not assign:
            set_to_idx = {}
            for _, (ids, vals) in self.dataframe[[f'{name}_ID', f'{name}_NAME']].iterrows():
                if redundant:
                    if len(ids) != len(vals):
                        raise ValueError('Row id/name length mismatch')
                    for i, v in zip(ids, vals):
                        if isnan(i) or (isinstance(v, float) and isnan(v)): continue
                        i = int(i)
                        if v in set_to_idx and set_to_idx[v] != i:
                            raise ValueError(f'Invalid {name} id {i} assigned to name {v}')
                        else: set_to_idx[v] = i
                else:
                    if isnan(ids) or (isinstance(vals, float) and isnan(vals)): continue
                    ids = int(ids)
                    if vals in set_to_idx and set_to_idx[vals] != ids:
                        raise ValueError(f'Invalid {name} id {ids} assigned to name {vals}')
                    else: set_to_idx[vals] = ids
            return set_to_idx, sorted(set_to_idx, key=set_to_idx.get)
        sets = set()
        default_value = ['default'] if redundant else 'default'
        if default:
            self.dataframe.loc[self.dataframe[f'{name}_NAME'].isna(), f'{name}_NAME'] = \
                self.dataframe.loc[self.dataframe[f'{name}_NAME'].isna(), f'{name}_NAME'].apply(
                    lambda x: default_value)
        for v in self.dataframe[f'{name}_NAME']:
            if redundant: sets.update(v)
            else: sets.add(v)
        set_to_idx = {v: i for i, v in enumerate(sets)}
        if redundant:
            self.dataframe[f'{name}_ID'] = self.dataframe[f'{name}_NAME'].apply(
                lambda x: list(map(lambda y: set_to_idx[y], x)))
        else:
            self.dataframe[f'{name}_ID'] = self.dataframe[f'{name}_NAME'].apply(
                lambda x: set_to_idx[x])
        return set_to_idx, sorted(set_to_idx, key=set_to_idx.get)

    def _patch_ids(self, name: str, set_to_idx: dict, idx_to_set: list, redundant=False) -> None:
        for i, (ids, vals) in self.dataframe[[f'{name}_ID', f'{name}_NAME']].iterrows():
            if redundant:
                for index, (k, v) in enumerate(zip(ids, vals)):
                    if isnan(i):
                        self.dataframe.at[k, f'{name}_ID'][index] = set_to_idx[v]
                    if isinstance(v, float) and isnan(v):
                        self.dataframe.at[k, f'{name}_NAME'][index] = idx_to_set[v]
            else:
                if isnan(ids):
                    self.dataframe.at[i, f'{name}_ID'] = set_to_idx[vals]
                if isinstance(vals, float) and isnan(vals):
                    self.dataframe.at[i, f'{name}_NAME'] = idx_to_set[ids]

    def cleanup(self) -> None:
        '''
        Run cleanup and sanity checks on all data.
        '''
        print('[CVData] Cleaning up data...')

        # convert bounding boxes into proper format and store under 'BOX'
        if {'X1', 'X2', 'Y1', 'Y2'}.issubset(self.dataframe.columns):
            self._convert_bbox(0)
        elif {'XMIN', 'YMIN', 'XMAX', 'YMAX'}.issubset(self.dataframe.columns):
            self._convert_bbox(1)
        elif {'XMIN', 'YMIN', 'WIDTH', 'HEIGHT'}.issubset(self.dataframe.columns):
            self._convert_bbox(2)

        # assign image ids
        if 'IMAGE_ID' not in self.dataframe: self.dataframe['IMAGE_ID'] = self.dataframe.index

        # assign class ids
        if 'CLASS_NAME' in self.dataframe:
            if 'CLASS_ID' not in self.dataframe:
                print('[CVData] Assigning CLASS_ID')
                self.class_to_idx, self.classes = self._assign_ids('CLASS', assign=True)
            else:
                self.class_to_idx, self.classes = self._assign_ids('CLASS', assign=False)
                self._patch_ids('CLASS', set_to_idx=self.class_to_idx, idx_to_set=self.classes)

        if 'SEG_CLASS_NAME' in self.dataframe:
            if 'SEG_CLASS_ID' not in self.dataframe:
                print('[CVData] Assigning SEG_CLASS_ID')
                self.seg_class_to_idx, self.seg_classes = self._assign_ids('SEG_CLASS', assign=True, 
                                                                   redundant=True)
            else:
                self.seg_class_to_idx, self.seg_classes = self._assign_ids('SEG_CLASS', redundant=True)
        elif 'SEG_CLASS_ID' in self.dataframe:
            self.seg_classes = set()
            for item in self.dataframe['SEG_CLASS_ID']: self.seg_classes.update(item)

        if CVData._classification_cols.issubset(self.dataframe.columns):
            self.available_modes.append('classification')
        if CVData._detection_cols.issubset(self.dataframe.columns):
            self.available_modes.append('detection')
        if CVData._segmentation_cols.issubset(self.dataframe.columns) and \
            ({'POLYGON', 'SEG_CLASS_ID'}.issubset(self.dataframe.columns) \
                or 'ABSOLUTE_FILE_SEG' in self.dataframe):
            self.available_modes.append('segmentation')
        # add segmentation mode
        self.dataframe.drop(columns='GENERIC', inplace=True, errors='ignore')
        self._cleanup_image_sets()
        self._cleanup_id()
        self.cleaned = True
        print('[CVData] Done!')

    def _convert_bbox(self, mode: int) -> None:
        boxes = []
        def __execute_checks(boxes: list, cols: tuple):
            if any([isinstance(row[cols[0]], float), isinstance(row[cols[1]], float),
                    isinstance(row[cols[2]], float), isinstance(row[cols[3]], float)]):
                boxes.append([])
                return False
            assert all(len(row[x]) == len(row[cols[0]]) for x in cols), \
                'Length of bbox lists does not match'
            return True
        if mode == 0:
            for _, row in self.dataframe.iterrows():
                if not __execute_checks(boxes, ('X1', 'Y1', 'X2', 'Y2')): continue
                boxes.append([(min(x1, x2), min(y1, y2), max(x1, x2), max(y1,y2)) for x1, y1, x2, y2
                              in zip(row['X1'], row['Y1'], row['X2'], row['Y2'])])
            self.dataframe['BOX'] = boxes
            return
        if mode == 1:
            for _, row in self.dataframe.iterrows():
                if not __execute_checks(boxes, ('XMIN', 'YMIN', 'XMAX', 'YMAX')): continue
                boxes.append([(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax
                              in zip(row['XMIN'], row['YMIN'], row['XMAX'], row['YMAX'])])
            self.dataframe['BOX'] = boxes
            return
        if mode == 2:
            for _, row in self.dataframe.iterrows():
                if not __execute_checks(boxes, ('XMIN', 'YMIN', 'WIDTH', 'HEIGHT')): continue
                boxes.append([(xmin, ymin, xmin+width, ymin+height) for xmin, ymin, width, height
                              in zip(row['XMAX'], row['YMAX'], row['WIDTH'], row['HEIGHT'])])
            self.dataframe['BOX'] = boxes
            return

    def _cleanup_id(self) -> None:
        cols = ['CLASS_ID', 'IMAGE_ID']
        for col in cols:
            if col not in self.dataframe: continue
            self.dataframe[col] = self.dataframe[col].astype('Int64')

    def _cleanup_image_sets(self) -> None:
        if 'IMAGE_SET_ID' not in self.dataframe:
            if 'IMAGE_SET_NAME' not in self.dataframe:
                print('[CVData] No image set found. Assigning to default image set')
                self.dataframe['IMAGE_SET_NAME'] = [['default']] * len(self.dataframe)
                self.dataframe['IMAGE_SET_ID'] = [[0]] * len(self.dataframe)
                self.image_set_to_idx = {'default': 0}
                self.image_sets = {'default'}
            else:
                print('[CVData] Converting IMAGE_SET names to IMAGE_SET_ID')
                self.image_set_to_idx, self.image_sets = self._assign_ids('IMAGE_SET',
                                                                          default=True,
                                                                          redundant=True,
                                                                          assign=True)
        elif 'IMAGE_SET_NAME' in self.dataframe:
            self.image_set_to_idx, self.image_sets = self._assign_ids('IMAGE_SET', default=True,
                                                                      redundant=True)
        else:
            for ids in self.dataframe['IMAGE_SET_ID']:
                self.image_sets.update(ids)

    def get_dataset(self, mode: str, image_set: str) -> Dataset:
        '''
        Retrieve the dataset.
        '''
        if not self.cleaned: self.cleanup()
        assert mode.lower().strip() in self.available_modes, 'Desired mode not available.'
        dataframe = self.dataframe[[image_set in item for item in self.dataframe['IMAGE_SET_NAME']]]
        seg_default = 0
        if len(dataframe) == 0: raise ValueError(f'Image set {image_set} not available.')
        if mode == 'classification':
            dataframe = dataframe[list(CVData._classification_cols)]
            if self.remove_invalid:
                print(f'Removed {len(dataframe[dataframe.isna().any(axis=1)])} NaN entries.')
                dataframe = dataframe.dropna()
        elif mode == 'detection':
            dataframe = dataframe[list(CVData._detection_cols)]
            if self.remove_invalid:
                print(f'Removed {len(dataframe[dataframe.isna().any(axis=1)])} NaN entries.')
                dataframe = dataframe.dropna()
        elif mode == 'segmentation':
            dataframe = dataframe[list(CVData._segmentation_cols) + (['POLYGON', 'SEG_CLASS_ID'] \
                if 'POLYGON' in dataframe else ['ABSOLUTE_FILE_SEG'])]
            if self.remove_invalid:
                print(f'Removed {len(dataframe[dataframe.isna().any(axis=1)])} NaN entries.')
                dataframe = dataframe.dropna()
            seg_default = max(self.seg_classes) + 1
        if len(dataframe) == 0: raise ValueError('[CVData] After cleanup, this dataset is empty.')
        return CVDataset(dataframe, mode, seg_default=seg_default)

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
    def __init__(self, df: DataFrame, mode: str, seg_default: int = 0):
        self.dataframe = df
        self.data = self.dataframe.to_dict('records')
        self.mode = mode
        self.seg_default = seg_default

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item: dict = self.data[idx]
        image: Tensor = read_image(item.get('ABSOLUTE_FILE'))
        label: dict[str, Tensor]
        if self.mode == 'classification':
            label = _get_class_labels(item)
        elif self.mode == 'detection':
            label = _get_bbox_labels(item)
        elif self.mode == 'segmentation':
            label = _get_seg_labels(item, default=self.seg_default)
        return image, label
