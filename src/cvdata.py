'''
Main module for processing datasets.
'''
import os
import time
import json
from typing import Any, Union, Optional, Callable
from typing_extensions import Self
from math import isnan
from numpy import asarray, int32, full_like, nan
import random
import jsonpickle
from pandas import DataFrame
from pandas.core.series import Series
from cv2 import imread, fillPoly, IMREAD_GRAYSCALE
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, LongTensor, FloatTensor
from torchvision.datasets.vision import VisionDataset
import torch
from PIL.Image import open as open_image
from PIL.Image import fromarray



from ._utils import next_avail_id
from .data_items import Static, Generic, DataTypes, DataType
from .populate import populate_data

def _collate_detection(batch):
    images, labels = zip(*batch)
    return tuple(images), tuple(labels)

def _collate_segmentation(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)

def _collate(mode: str) -> Callable:
    if mode == 'segmentation': return _collate_segmentation
    if mode == 'detection': return _collate_detection

class CVData:
    '''
    Main dataset class utils.
    '''
    _classification_cols = {'ABSOLUTE_FILE', 'IMAGE_ID', 'CLASS_ID'}
    _detection_cols = {'ABSOLUTE_FILE', 'IMAGE_ID', 'BBOX_CLASS_ID', 'BOX'}
    _segmentation_img_cols = {'ABSOLUTE_FILE', 'IMAGE_ID', 'ABSOLUTE_FILE_SEG'}
    _segmentation_poly_cols = {'ABSOLUTE_FILE', 'IMAGE_ID', 'POLYGON', 'SEG_CLASS_ID'}

    def __init__(
        self, 
        root: str, 
        form: dict[Union[Static, Generic], Any],
        remove_invalid: bool = True
    ) -> None:
        self.root = root
        self.form = form
        self.image_set_to_idx = {}
        self.idx_to_image_set = {}
        self.seg_class_to_idx = {}
        self.idx_to_seg_class = {}
        self.bbox_class_to_idx = {}
        self.idx_to_bbox_class = {}
        self.available_modes = []
        self.cleaned = False
        self.remove_invalid = remove_invalid

    def parse(self, override: bool = False) -> None:
        print('[CVData] Parsing data...')
        start = time.time()
        if self.cleaned and not override:
            raise ValueError('Dataset has already been parsed. Use override=True to override')
        data = populate_data(self.root, self.form)
        entries = [{key: val.value for key, val in item.data.items()} for item in data]
        self.dataframe = DataFrame(entries)
        end = time.time()
        print(f'[CVData] Parsed! ({end - start}s)')
        print('[CVData] Cleaning up...')
        start = time.time()
        self.cleanup()
        end = time.time()
        print(f'[CVData] Cleaned! ({end - start}s)')
        

    def cleanup(self) -> None:
        '''
        Run cleanup and sanity checks on all data.
        '''
        print('[CVData] Cleaning up data...')

        # sort by image id first to prevent randomness
        if 'IMAGE_ID' in self.dataframe:
            self.dataframe.sort_values('IMAGE_ID', ignore_index=True, inplace=True)
        else:
            self.dataframe.sort_values('IMAGE_NAME', ignore_index=True, inplace=True)

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
                self.class_to_idx, self.idx_to_class = self._assign_ids('CLASS')
            else:
                self.class_to_idx, self.idx_to_class = self._validate_ids('CLASS')
                self._patch_ids('CLASS', name_to_idx=self.class_to_idx, idx_to_name=self.idx_to_class)

        # assign seg ids
        if 'SEG_CLASS_NAME' in self.dataframe:
            if 'SEG_CLASS_ID' not in self.dataframe: call = self._assign_ids
            else: call = self._validate_ids
            result = call('SEG_CLASS', redundant=True)
            self.seg_class_to_idx, self.idx_to_seg_class = result
        elif 'SEG_CLASS_ID' in self.dataframe:
            self.idx_to_seg_class = {i: str(i) for item in self.dataframe['SEG_CLASS_ID'] for i in item}
            self.seg_class_to_idx = {str(i): i for item in self.dataframe['SEG_CLASS_ID'] for i in item}

        # assign bbox ids
        if 'BBOX_CLASS_NAME' in self.dataframe:
            if 'BBOX_CLASS_ID' not in self.dataframe: call = self._assign_ids
            else: call = self._validate_ids
            result = call('BBOX_CLASS', redundant=True)
            self.bbox_class_to_idx, self.idx_to_bbox_class = result
        elif 'BBOX_CLASS_ID' in self.dataframe:
            self.idx_to_bbox_class = {i: str(i) for item in self.dataframe['BBOX_CLASS_ID'] for i in item}
            self.bbox_class_to_idx = {str(i): i for item in self.dataframe['BBOX_CLASS_ID'] for i in item}

        # check available columns to determine mode availability
        if CVData._classification_cols.issubset(self.dataframe.columns):
            self.available_modes.append('classification')
        if CVData._detection_cols.issubset(self.dataframe.columns):
            self.available_modes.append('detection')
        if CVData._segmentation_img_cols.issubset(self.dataframe.columns) or \
            CVData._segmentation_poly_cols.issubset(self.dataframe.columns):
            self.available_modes.append('segmentation')
        
        # cleanup image sets
        self.dataframe.drop(columns='GENERIC', inplace=True, errors='ignore')
        self._cleanup_image_sets()
        self._cleanup_id()
        self.cleaned = True
        print('[CVData] Done!')

    def _validate_ids(self, name: str, redundant=False) -> tuple[dict[str, int], dict[int, str]]:
        def check(i: int, v: str, name_to_idx: dict[str, int]) -> None:
            '''Check whether a value is corrupted/mismatch, and update dict accordingly'''
            if isnan(i) or (isinstance(v, float) and isnan(v)): return
            i = int(i)
            if v in name_to_idx and name_to_idx[v] != i:
                raise ValueError(f'Invalid {name} id {i} assigned to name {v}')
            else: name_to_idx[v] = i
        # validate all values and populate name_to_idx if assign is off=
        name_to_idx = {}
        for i, (ids, vals) in self.dataframe[[f'{name}_ID', f'{name}_NAME']].iterrows():
            if (isinstance(ids, float) and isnan(ids)) or (isinstance(vals, float) and isnan(vals)):
                continue
            if redundant:
                if len(ids) != len(vals): raise ValueError(f'ID/name mismatch at row {i}')
                for i, v in zip(ids, vals): check(i, v, name_to_idx)
            else: check(ids, vals, name_to_idx)
        return name_to_idx, {v: k for k, v in name_to_idx.items()}

    def _assign_ids(self, name: str, default=False, redundant=False) -> \
            tuple[dict[str, int], dict[int, str]]:
        sets = set()
        default_value = ['default'] if redundant else 'default'
        if default:
            self.dataframe.loc[self.dataframe[f'{name}_NAME'].isna(), f'{name}_NAME'] = \
                self.dataframe.loc[self.dataframe[f'{name}_NAME'].isna(), f'{name}_NAME'].apply(
                    lambda x: default_value)
        for v in self.dataframe[f'{name}_NAME']:
            if isinstance(v, float): continue
            if redundant: sets.update(v)
            else: sets.add(v)
        name_to_idx = {v: i for i, v in enumerate(sets)}
        idx_to_name = {v: k for k, v in name_to_idx.items()}
        if redundant:
            self.dataframe[f'{name}_ID'] = self.dataframe[f'{name}_NAME'].apply(
                lambda x: nan if isinstance(x, float) else list(map(lambda y: name_to_idx[y], x)))
        else:
            self.dataframe[f'{name}_ID'] = self.dataframe[f'{name}_NAME'].apply(
                lambda x: nan if isinstance(x, float) else name_to_idx[x])
        return name_to_idx, idx_to_name

    def _patch_ids(self, name: str, name_to_idx: dict, idx_to_name: dict, redundant=False) -> None:
        '''Patch nan values of ids/vals accordingly.'''
        if not redundant:
            for i, (ids, vals) in self.dataframe[[f'{name}_ID', f'{name}_NAME']].iterrows():
                if isnan(ids) and isinstance(vals, float) and isnan(vals): 
                    print(f'Found missing {name} id/name at row {i}')
                    continue
                if isnan(ids): self.dataframe.at[i, f'{name}_ID'] = name_to_idx[vals]
                if isinstance(vals, float) and isnan(vals):
                    self.dataframe.at[i, f'{name}_NAME'] = idx_to_name[ids]
            return
        for i, (ids, vals) in self.dataframe[[f'{name}_ID', f'{name}_NAME']].iterrows():
            for index, (k, v) in enumerate(zip(ids, vals)):
                print(f'Found missing {name} id/name at row {i}')
                if isnan(k) and isinstance(v, float) and isnan(v):
                    print(f'Found missing {name} id/name at row {i}')
                    continue
                if isnan(k): self.dataframe.at[i, f'{name}_ID'][index] = name_to_idx[v]
                if isinstance(v, float) and isnan(v):
                    self.dataframe.at[i, f'{name}_NAME'][index] = idx_to_name[v]

    def _convert_bbox(self, mode: int) -> None:
        boxes = []
        def execute_checks(boxes: list, cols: tuple):
            if any([isinstance(row[cols[0]], float), isinstance(row[cols[1]], float),
                    isinstance(row[cols[2]], float), isinstance(row[cols[3]], float)]):
                boxes.append([])
                return False
            assert all(len(row[x]) == len(row[cols[0]]) for x in cols), \
                'Length of bbox lists does not match'
            return True
        if mode == 0:
            for _, row in self.dataframe.iterrows():
                if not execute_checks(boxes, ('X1', 'Y1', 'X2', 'Y2')): continue
                boxes.append([(min(x1, x2), min(y1, y2), max(x1, x2), max(y1,y2)) for x1, y1, x2, y2
                              in zip(row['X1'], row['Y1'], row['X2'], row['Y2'])])
        elif mode == 1:
            for _, row in self.dataframe.iterrows():
                if not execute_checks(boxes, ('XMIN', 'YMIN', 'XMAX', 'YMAX')): continue
                boxes.append([(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax
                              in zip(row['XMIN'], row['YMIN'], row['XMAX'], row['YMAX'])])
        elif mode == 2:
            for _, row in self.dataframe.iterrows():
                if not execute_checks(boxes, ('XMIN', 'YMIN', 'WIDTH', 'HEIGHT')): continue
                boxes.append([(xmin, ymin, xmin+width, ymin+height) for xmin, ymin, width, height
                              in zip(row['XMAX'], row['YMAX'], row['WIDTH'], row['HEIGHT'])])
        self.dataframe['BOX'] = boxes

    def _cleanup_id(self) -> None:
        cols = ['CLASS_ID', 'IMAGE_ID']
        for col in cols:
            if col not in self.dataframe: continue
            self.dataframe[col] = self.dataframe[col].astype('Int64')

    def _cleanup_image_sets(self) -> None:
        if 'IMAGE_SET_NAME' in self.dataframe:
            result = self._assign_ids('IMAGE_SET', default=True, redundant=True)
            self.image_set_to_idx, self.idx_to_image_set = result
        elif 'IMAGE_SET_ID' not in self.dataframe:
            self.dataframe['IMAGE_SET_NAME'] = [['default']] * len(self.dataframe)
            self.dataframe['IMAGE_SET_ID'] = [[0]] * len(self.dataframe)
            self.image_set_to_idx = {'default': 0}
            self.idx_to_image_set = {0: 'default'}
        else:
            for ids in self.dataframe['IMAGE_SET_ID']:
                self.idx_to_image_set.update({k: str(k) for k in ids})
                self.image_set_to_idx.update({str(k): k for k in ids})

    def get_dataset(
        self, 
        mode: str, 
        image_set: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[tuple[Callable]] = None
    ) -> Dataset:
        '''
        Retrieve the dataset.
        '''
        if transforms: transform, target_transform = transforms
        if not self.cleaned: self.cleanup()
        assert mode.lower().strip() in self.available_modes, 'Desired mode not available.'
        dataframe = self.dataframe[[image_set in item for item in self.dataframe['IMAGE_SET_NAME']]]
        if len(dataframe) == 0: raise ValueError(f'Image set {image_set} not available.')
        if mode == 'classification': 
            dataframe = dataframe[list(CVData._classification_cols)]
            id_mapping = {k: i for i, k in enumerate(self.idx_to_class)}
        elif mode == 'detection': 
            dataframe = dataframe[list(CVData._detection_cols)]
            id_mapping = {k: i for i, k in enumerate(self.idx_to_bbox_class)}
        elif mode == 'segmentation':
            dataframe = dataframe[list(CVData._segmentation_poly_cols if 'POLYGON' in dataframe 
                                       else CVData._segmentation_img_cols)]
            id_mapping = {k: i for i, k in enumerate(self.idx_to_seg_class)}
        if self.remove_invalid:
            print(f'Removed {len(dataframe[dataframe.isna().any(axis=1)])} NaN entries.')
            dataframe = dataframe.dropna()
        if len(dataframe) == 0: raise ValueError('[CVData] After cleanup, this dataset is empty.')
        return CVDataset(dataframe, self.root, mode, id_mapping=id_mapping, transform=transform,
                         target_transform=target_transform)

    def get_dataloader(
        self,
        mode: str,
        image_set: str,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 1,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[tuple[Callable]] = None
    ) -> DataLoader:
        '''
        Retrieve the dataloader for this dataset.
        '''
        return DataLoader(
            self.get_dataset(
                mode,
                image_set,
                transform=transform,
                target_transform=target_transform,
                transforms=transforms),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=_collate(mode)
        )
    
    def split_image_set(
        self, image_set: str,
        *new_sets: tuple[str, float],
        inplace: bool = False,
        seed: int = None
    ) -> None:
        '''
        Split the existing image set into new image sets.
        '''
        # checks before splitting
        if image_set not in self.image_set_to_idx: 
            raise ValueError("Image set doesn't exist!")
        if any(new_set[0] in self.image_set_to_idx for new_set in new_sets): 
            raise ValueError(f'New set name already exists!')
        tot_frac = sum(new_set[1] for new_set in new_sets)
        if not inplace and tot_frac != 1:
            raise ValueError(f'Split fraction invalid, not equal to 1')
        if inplace and tot_frac > 1:
            raise ValueError(f'Split fraction invalid, greater than 1')

        # assemble new sets
        new_sets: list = list(new_sets)
        if inplace: new_sets.append((image_set, 1 - tot_frac))
        if seed is not None: random.seed(seed)

        # add to existing image set tracker 
        for k in new_sets:
            next_id = next_avail_id(self.idx_to_image_set)
            self.idx_to_image_set[next_id] = k[0]
            self.image_set_to_idx[k[0]] = next_id

        # assign image sets
        for _, row in self.dataframe.iterrows():
            if image_set not in row['IMAGE_SET_NAME']: continue
            partition = random.random()
            running_sum = 0
            for _, next_set in enumerate(new_sets):
                if running_sum <= partition <= running_sum + next_set[1]:
                    if inplace:
                        index = row['IMAGE_SET_NAME'].index(image_set)
                        row['IMAGE_SET_NAME'][index] = next_set[0]
                        row['IMAGE_SET_ID'][index] = self.image_set_to_idx[next_set[0]]
                    else: 
                        row['IMAGE_SET_NAME'].append(next_set[0])
                        row['IMAGE_SET_ID'].append(self.image_set_to_idx[next_set[0]])
                    break
                running_sum += next_set[1]
        if inplace: self.clear_image_sets()
    
    def get_image_set(
        self,
        image_set: Union[str, int]
    ) -> DataFrame:
        if isinstance(image_set, str):
            return self.dataframe[self.dataframe['IMAGE_SET_NAME'].apply(lambda x: image_set in x)]
        return self.dataframe[self.dataframe['IMAGE_SET_ID'].apply(lambda x: image_set in x)]
    
    def clear_image_sets(
        self
    ) -> None:
        to_pop = []
        for image_set in self.image_set_to_idx:
            if len(self.get_image_set(image_set)) == 0:
                to_pop.append(image_set)
        for image_set in to_pop:
            index = self.image_set_to_idx.pop(image_set)
            self.idx_to_image_set.pop(index)
    
    def delete_image_set(
        self,
        image_set: Union[str, int]
    ) -> None:
        using_id: bool = isinstance(image_set, int)
        if using_id:
            if image_set not in self.idx_to_image_set: 
                raise KeyError(f'Invalid ID: {image_set}')
            idx = image_set
            name = self.idx_to_image_set[idx]
        else:
            if image_set not in self.image_set_to_idx: 
                raise KeyError(f'Invalid name: {image_set}')
            name = image_set
            idx = self.image_set_to_idx[name]

        default = False
        if 'default' in self.image_set_to_idx:
            default_idx = self.image_set_to_idx['default'] 
        else:   
            default_idx = next_avail_id(self.idx_to_image_set)

        for _, row in self.dataframe.iterrows():
            if idx in row['IMAGE_SET_ID']:
                row['IMAGE_SET_ID'].remove(idx)
                row['IMAGE_SET_NAME'].remove(name)
                if len(row['IMAGE_SET_ID']) == 0:
                    row['IMAGE_SET_ID'].append(default_idx)
                    row['IMAGE_SET_NAME'].append('default')
                    default = True

        if default and 'default' not in self.image_set_to_idx:
            self.image_set_to_idx['default'] = default_idx
            self.idx_to_image_set[default_idx] = 'default'
        
        self.clear_image_sets()

    def save(
        self,
        filename: str,
        overwrite: bool = False
    ) -> None:
        '''
        Save the dataset into json format.
        '''
        this = {
            'root': self.root,
            'form': jsonpickle.encode(self.form, keys=True),
            'dataframe': self.dataframe.to_json(),
            'image_set_to_idx': self.image_set_to_idx,
            'idx_to_image_set': self.idx_to_image_set,
            'seg_class_to_idx': self.seg_class_to_idx,
            'idx_to_seg_class': self.idx_to_seg_class,
            'bbox_class_to_idx': self.bbox_class_to_idx,
            'idx_to_bbox_class': self.idx_to_bbox_class,
            'remove_invalid': self.remove_invalid,
            'available_modes': self.available_modes,
            'cleaned': self.cleaned,
        }
        if os.path.exists(filename) and not overwrite:
            raise ValueError(f'File already exists: {filename}')
        with open(filename, 'w') as f:
            f.write(json.dumps(this))

    @classmethod
    def load(cls, filename: str) -> Self:
        try:
            print('[CVData] Loading dataset...')
            start = time.time()
            with open(filename, 'r') as f:
                data = json.load(f)
            this: CVData = cls(data['root'], jsonpickle.decode(data['form'], keys=True))
            this.dataframe = DataFrame.from_dict(json.loads(data['dataframe']))
            this.image_set_to_idx = data['image_set_to_idx']
            this.idx_to_image_set = data['idx_to_image_set']
            this.seg_class_to_idx = data['seg_class_to_idx']
            this.idx_to_seg_class = data['idx_to_seg_class']
            this.bbox_class_to_idx = data['bbox_class_to_idx']
            this.idx_to_bbox_class = data['idx_to_bbox_class']
            this.remove_invalid = data['remove_invalid']
            this.available_modes = data['available_modes']
            this.cleaned = data['cleaned']
            end = time.time()
            print(f'[CVData] Loaded dataset! ({end - start}s)')
        except Exception:
            print('[CVData] This is not a CVData dataset!')
            return None
        return this
        
class CVDataset(VisionDataset):
    '''
    Dataset implementation for CVData environment.
    '''
    def __init__(
        self,
        df: DataFrame,
        root: str,
        mode: str,
        id_mapping: dict[int, int],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_type: str = 'RGB'
    ):
        self.dataframe = df
        self.data = self.dataframe.to_dict('records')
        self.mode = mode
        self.image_type = image_type
        self.id_mapping = id_mapping
        if self.mode == 'segmentation': self.default = len(self.id_mapping)
        super().__init__(root, transforms=None, transform=transform, target_transform=target_transform)

    def __len__(self):
        return len(self.dataframe)

    def _get_class_labels(self, item: Series) -> Tensor:
        return self.id_mapping[int(item['CLASS_ID'])]

    def _get_bbox_labels(self, item: Series) -> dict[str, Tensor]:
        # execute checks
        assert len(item['BOX']) == len(item['BBOX_CLASS_ID']), \
            'SEG_CLASS_ID and POLYGON len mismatch'
        class_ids = list(map(lambda x: self.id_mapping[x], item['BBOX_CLASS_ID']))
        boxes = item['BOX']
        bbox_tensors = [FloatTensor(box) for box in boxes]
        return {'boxes': torch.stack(bbox_tensors), 'labels': LongTensor(class_ids)}

    def _get_seg_labels(self, item: Series) -> Tensor:
        if 'ABSOLUTE_FILE_SEG' in item:
            return open_image(item['ABSOLUTE_FILE_SEG'])
        assert len(item['POLYGON']) == len(item['SEG_CLASS_ID']), \
            'SEG_CLASS_ID and POLYGON len mismatch'
        mask = asarray(imread(item['ABSOLUTE_FILE'], IMREAD_GRAYSCALE), dtype=int32)
        mask = full_like(mask, self.default)
        for class_id, polygon in zip(item['SEG_CLASS_ID'], item['POLYGON']):
            mask = fillPoly(mask, pts=[asarray(polygon, dtype=int32)],
                            color=self.id_mapping[class_id])
        mask = fromarray(asarray(mask))
        return mask

    def __getitem__(self, idx):
        item: dict = self.data[idx]
        image: Tensor = open_image(item.get('ABSOLUTE_FILE')).convert(self.image_type)
        label: dict[str, Tensor]
        if self.mode == 'classification':
            label = self._get_class_labels(item)
        elif self.mode == 'detection':
            label = self._get_bbox_labels(item)
        elif self.mode == 'segmentation':
            label = self._get_seg_labels(item)
        if self.transforms: image, label = self.transforms(image, label)
        return image, label
