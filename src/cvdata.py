'''
Main module for processing datasets. Collects all parsed objects into the CVData object, and
maintains the CVDataset class for PyTorch Dataset and DataLoader functionalities.
'''
import os
import time
import json
from typing import Union, Optional, Callable, Any
from typing_extensions import Self
from math import isnan
from numpy import asarray, full_like, full, nan, int32
import random
import jsonpickle
from pandas import DataFrame
from pandas.core.series import Series
from cv2 import imread, fillPoly, IMREAD_GRAYSCALE
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, LongTensor, FloatTensor
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from hashlib import md5
from PIL.Image import open as open_image
import matplotlib.pyplot as plt

from ._utils import next_avail_id, union
from .processing import populate_data

def _collate_detection(batch):
    images, labels = zip(*batch)
    return tuple(images), tuple(labels)

def _collate_segmentation_dim(batch):
    images, labels = zip(*batch)
    labels = {'label': torch.stack([label['label'] for label in labels]),
              'dim': [label['dim'] for label in labels]}
    return torch.stack(images), labels

def _collate_segmentation(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)

def _collate_classification_dim(batch):
    images, labels = zip(*batch)
    labels = {'label': torch.IntTensor([label['label'] for label in labels]),
              'dim': [label['dim'] for label in labels]}
    return torch.stack(images), labels

def _collate_default(batch):
    return list(zip(*batch))

def _collate(mode: str, store_dim: bool) -> Callable:
    if mode == 'segmentation':
        return _collate_segmentation_dim if store_dim else _collate_segmentation
    if mode == 'detection':
        return _collate_detection
    if mode == 'classification' and store_dim:
        return _collate_classification_dim
    if mode == 'inference' or mode == 'diffusion':
        return _collate_default

class CVData:
    '''
    Main dataset class. Accepts root directory path and dictionary form of the structure.
    
    Args:
    - root (str): the root directory to access the dataset.
    - form (dict): the form of the dataset. See documentation for further details on valid forms.
    - get_md5_hashes (bool): when set to True, create a new column which finds md5 hashes for each
                             image available, and makes sure there are no duplicates. Default: False
    - bbox_scale_option (str): choose from either 'auto', 'zeroone', or 'full' scale options to
                               define, or leave empty for automatic. Default: 'auto'
    - seg_scale_option (str): choose from either 'auto', 'zeroone', or 'full' scale options to
                              define, or leave empty for automatic. Default: 'auto'
    '''

    _inference_cols = {'ABSOLUTE_FILE', 'IMAGE_ID', 'IMAGE_DIM'}
    _diffusion_cols = {'ABSOLUTE_FILE', 'IMAGE_ID', 'IMAGE_DIM'}
    _classification_cols = {'ABSOLUTE_FILE', 'IMAGE_ID', 'CLASS_ID', 'IMAGE_DIM'}
    _detection_cols = {'ABSOLUTE_FILE', 'IMAGE_ID', 'BBOX_CLASS_ID', 'BOX', 'IMAGE_DIM'}
    _segmentation_img_cols = {'ABSOLUTE_FILE', 'IMAGE_ID', 'ABSOLUTE_FILE_SEG', 'IMAGE_DIM'}
    _segmentation_poly_cols = {'ABSOLUTE_FILE', 'IMAGE_ID', 'POLYGON', 'SEG_CLASS_ID', 'IMAGE_DIM'}

    def __init__(
        self, 
        root: str, 
        form: dict,
        get_md5_hashes: bool = False,
        bbox_scale_option: str = 'auto',
        seg_scale_option: str = 'auto'
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
        self.get_img_dim = True
        self.get_md5_hashes = get_md5_hashes
        self.bbox_scale_option = bbox_scale_option
        self.seg_scale_option = seg_scale_option

    def parse(self, override: bool = False) -> None:
        '''
        Must be called to instantiate the data in the dataset instance. Performs the recursive
        populate_data algorithm and creates the dataframe, and then cleans up the data.
        
        Args:
        - override (bool): whether to overwrite existing data if it has already been parsed and
                           cleaned. Default: False
        '''
        print('[CVData] Parsing data...')
        start = time.time()
        if self.cleaned and not override:
            raise ValueError('Dataset has already been parsed. Use override=True to override')
        data = populate_data(self.root, self.form)
        entries = [{key: val.value for key, val in item.data.items()} for item in data]
        self.dataframe = DataFrame(entries)
        end = time.time()
        print(f'[CVData] Parsed! ({(end - start):.3f}s)')
        start = time.time()
        self.cleanup()
        end = time.time()
        print(f'[CVData] Done! ({(end - start):.3f}s)')
        print(self._get_statistics())
        
    def _get_statistics(self):
        data = f'[CVData] Dataset statistics:\n'
        data += f'       | Available modes: {", ".join(self.available_modes)}\n'
        data += f'       | Images: {len(self.dataframe)}\n'
        for mode in self.available_modes:
            if mode != 'segmentation':
                data += f'       | Complete entries for {mode}: {len(self.dataframe)-len(self.dataframe[self.dataframe[list(getattr(self, f"_{mode}_cols"))].isna().any(axis=1)])}\n'
                continue
            if 'POLYGON' in self.dataframe.columns:
                data += f'       | Complete entries for {mode}: {len(self.dataframe)-len(self.dataframe[self.dataframe[list(getattr(self, f"_{mode}_poly_cols"))].isna().any(axis=1)])}\n'
            else:
                data += f'       | Complete entries for {mode}: {len(self.dataframe)-len(self.dataframe[self.dataframe[list(getattr(self, f"_{mode}_img_cols"))].isna().any(axis=1)])}\n'
            
        if 'detection' in self.available_modes:
            data += f'       | Bounding box coordinate scaling option: {self.bbox_scale_option}\n'
        if 'segmentation' in self.available_modes:
            data += f'       | Segmentation object coordinate scaling option: {self.seg_scale_option}\n'
        return data.strip()

    def cleanup(self) -> None:
        '''
        Run cleanup and sanity checks on all data. Assigns IDs to name-only values.
        '''
        print('[CVData] Cleaning up data...')

        # sort by image id first to prevent randomness
        if 'IMAGE_ID' in self.dataframe:
            self.dataframe.sort_values('IMAGE_ID', ignore_index=True, inplace=True)
        else:
            self.dataframe.sort_values('IMAGE_NAME', ignore_index=True, inplace=True)
            self.dataframe['IMAGE_ID'] = self.dataframe.index

        # get image sizes
        if self.get_img_dim: self._get_img_sizes()

        # get md5 hashes
        if self.get_md5_hashes: self._get_md5_hashes()

        # convert bounding boxes into proper format and store under 'BOX'
        if {'X1', 'X2', 'Y1', 'Y2'}.issubset(self.dataframe.columns):
            self._convert_bbox(0)
        elif {'XMIN', 'YMIN', 'XMAX', 'YMAX'}.issubset(self.dataframe.columns):
            self._convert_bbox(1)
        elif {'XMIN', 'YMIN', 'WIDTH', 'HEIGHT'}.issubset(self.dataframe.columns):
            self._convert_bbox(2)
        elif {'XMID', 'YMID', 'WIDTH', 'HEIGHT'}.issubset(self.dataframe.columns):
            self._convert_bbox(3)
        elif {'XMAX', 'YMAX', 'WIDTH', 'HEIGHT'}.issubset(self.dataframe.columns):
            self._convert_bbox(4)

        if 'BOX' in self.dataframe:
            self._get_box_scale()
            self._convert_box_scale()
        
        if 'POLYGON' in self.dataframe:
            self._get_seg_scale()
            self._convert_seg_scale()

        # assign class ids
        if 'CLASS_NAME' in self.dataframe:
            if 'CLASS_ID' not in self.dataframe: call = self._assign_ids
            else: call = self._validate_ids
            result = call('CLASS', redundant=False)
            self.class_to_idx, self.idx_to_class = result
        elif 'CLASS_ID' in self.dataframe:
            self.idx_to_class = {i: str(i) for item in self.dataframe['CLASS_ID']
                                 if not isinstance(item, float) for i in item}
            self.class_to_idx = {str(i): i for item in self.dataframe['CLASS_ID']
                                 if not isinstance(item, float) for i in item}
            names = [str(i) if not isinstance(i, float)
                     else nan for i in self.dataframe['CLASS_ID']]
            self.dataframe['CLASS_NAME'] = names

        # assign seg ids
        if 'SEG_CLASS_NAME' in self.dataframe:
            if 'SEG_CLASS_ID' not in self.dataframe: call = self._assign_ids
            else: call = self._validate_ids
            result = call('SEG_CLASS', redundant=True)
            self.seg_class_to_idx, self.idx_to_seg_class = result
        elif 'SEG_CLASS_ID' in self.dataframe:
            self.idx_to_seg_class = {i: str(i) for item in self.dataframe['SEG_CLASS_ID']
                                     if isinstance(item, list) for i in item}
            self.seg_class_to_idx = {str(i): i for item in self.dataframe['SEG_CLASS_ID']
                                     if isinstance(item, list) for i in item}
            names = [list(map(lambda x: self.idx_to_bbox_class[x], i)) if isinstance(i, list)
                     else [] for i in self.dataframe['SEG_CLASS_ID']]
            self.dataframe['SEG_CLASS_NAME'] = names

        # assign bbox ids
        if 'BBOX_CLASS_NAME' in self.dataframe:
            if 'BBOX_CLASS_ID' not in self.dataframe: call = self._assign_ids
            else: call = self._validate_ids
            result = call('BBOX_CLASS', redundant=True)
            self.bbox_class_to_idx, self.idx_to_bbox_class = result
        elif 'BBOX_CLASS_ID' in self.dataframe:
            self.idx_to_bbox_class = {i: str(i) for item in self.dataframe['BBOX_CLASS_ID']
                                      if isinstance(item, list) for i in item}
            self.bbox_class_to_idx = {str(i): i for item in self.dataframe['BBOX_CLASS_ID']
                                      if isinstance(item, list) for i in item}
            names = [list(map(lambda x: self.idx_to_bbox_class[x], i)) if isinstance(i, list)
                     else [] for i in self.dataframe['BBOX_CLASS_ID']]
            self.dataframe['BBOX_CLASS_NAME'] = names

        # check available columns to determine mode availability
        if CVData._inference_cols.issubset(self.dataframe.columns):
            self.available_modes.append('inference')
        if CVData._diffusion_cols.issubset(self.dataframe.columns):
            self.available_modes.append('diffusion')
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

    def _get_img_sizes(self) -> None:
        self.dataframe['IMAGE_DIM'] = [open_image(filename).size if isinstance(filename, str)
                                       else nan for filename in self.dataframe['ABSOLUTE_FILE']]

    def _get_md5_hashes(self) -> None:
        hashes = [md5(open_image(item).tobytes()) for item in self.dataframe['ABSOLUTE_FILE']]
        counter = {}
        for i, md5hash in enumerate(hashes):
            counter[md5hash] = counter.get(md5hash, []) + [i]
        duplicates = (locs for locs in counter.values() if len(locs) > 1)
        for locs in duplicates:
            raise ValueError(f'Found equivalent md5-hash images in the dataset at indices {locs}')
        self.dataframe['MD5'] = hashes

    def _get_box_scale(self) -> None:
        if self.bbox_scale_option == 'auto':
            for boxes in self.dataframe['BOX']:
                if any(coord > 1 for box in boxes for coord in box):
                    self.bbox_scale_option = 'full'
                    print('[CVData] Detected full size bounding box scale option')
                    return
                if any(coord < 0 for box in boxes for coord in box):
                    raise ValueError('[CVData] Detected unknown bounding box scale option')
        print('[CVData] Detected [0, 1] bounding box scale option')
        self.bbox_scale_option = 'zeroone'
        return

    def _get_seg_scale(self) -> None:
        if self.seg_scale_option == 'auto':
            for shapes in self.dataframe['POLYGON']:
                if any(val > 1 for shape in shapes for coord in shape for val in coord):
                    self.seg_scale_option = 'full'
                    print('[CVData] Detected full size bounding box scale option')
                    return
                if any(coord < 0 for shape in shapes for coord in shape):
                    raise ValueError('[CVData] Detected unknown bounding box scale option')
        print('[CVData] Detected [0, 1] bounding box scale option')
        self.seg_scale_option = 'zeroone'
        return

    def _convert_box_scale(self) -> None:
        if not self.get_img_dim:
            self.get_img_dim = True
            self._get_img_sizes()
        if self.bbox_scale_option == 'zeroone':
            boxes_list = []
            for _, row in self.dataframe[['BOX', 'IMAGE_DIM']].iterrows():
                if any(row.isna()):
                    boxes_list.append([])
                    continue
                apply_resize = lambda p: (p[0] * row['IMAGE_DIM'][0], p[1] * row['IMAGE_DIM'][1],
                                          p[2] * row['IMAGE_DIM'][0], p[3] * row['IMAGE_DIM'][1])
                boxes_list.append(list(map(apply_resize, row['BOX'])))
            self.dataframe['BOX'] = boxes_list
    
    def _convert_seg_scale(self) -> None:
        if not self.get_img_dim:
            self.get_img_dim = True
            self._get_img_sizes()
        if self.seg_scale_option == 'zeroone':
            shapes_list = []
            for _, row in self.dataframe[['POLYGON', 'IMAGE_DIM']].iterrows():
                if any(row.isna()):
                    shapes_list.append([])
                    continue
                apply_resize = lambda p: (p[0] * row['IMAGE_DIM'][0], p[1] * row['IMAGE_DIM'][1])
                shapes_list.append([list(map(apply_resize, shape)) for shape in row['POLYGON']])
            self.dataframe['POLYGON'] = shapes_list

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
        def execute_checks(i: int, row: Series, cols: tuple):
            if any(isinstance(row[cols[i]], float) for i in range(4)): return False
            if not all(len(row[x]) == len(row[cols[0]]) for x in cols):
                raise ValueError(f'Length of bbox lists at index {i} does not match ({len(row[cols[0]])}, {len(row[cols[1]])}, {len(row[cols[2]])}, {len(row[cols[3]])})')
            return True
        if mode == 0:
            cols = ('X1', 'Y1', 'X2', 'Y2')
            funcs = (min, min, max, max)
        elif mode == 1:
            cols = ('XMIN', 'YMIN', 'XMAX', 'YMAX')
            funcs = (lambda x: x[0], lambda y: y[0], lambda x: x[1], lambda y: y[1])
        elif mode == 2:
            cols = ('XMIN', 'YMIN', 'WIDTH', 'HEIGHT')
            funcs = (lambda x: x[0], lambda y: y[0], lambda x: x[0]+x[1], lambda y: y[0]+y[1])
        elif mode == 3:
            cols = ('XMID', 'YMID', 'WIDTH', 'HEIGHT')
            funcs = (lambda x: x[0]-x[1]/2, lambda y: y[0]-y[1]/2, lambda x: x[0]+x[1]/2, lambda y: y[0]+y[1]/2)
        elif mode == 4:
            cols = ('XMAX', 'YMAX', 'WIDTH', 'HEIGHT')
            funcs = (lambda x: x[0]-x[1], lambda y: y[0]-y[1], lambda x: x[0], lambda y: y[0])
        for i, row in self.dataframe.iterrows():
            if not execute_checks(i, row, cols):
                boxes.append([])
            else: boxes.append([(funcs[0]((x1, x2)), funcs[1]((y1, y2)), funcs[2]((x1, x2)), funcs[3]((y1, y2)))
                                for x1, y1, x2, y2 in zip(row[cols[0]], row[cols[1]], row[cols[2]], row[cols[3]])])
        self.dataframe['BOX'] = boxes
        self.dataframe.drop(['X1', 'Y1', 'X2', 'Y2', 'XMIN', 'YMIN', 'XMAX', 'YMAX', 'XMID', 'YMID', 'WIDTH', 'HEIGHT'], axis=1, inplace=True, errors='ignore')

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
        remove_invalid: bool = True,
        store_dim: bool = False,
        image_set: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[tuple[Callable]] = None,
        resize: Optional[tuple[int, int]] = None,
        normalize: Optional[str] = None
    ) -> Dataset:
        '''
        Retrieve the PyTorch dataset (torch.utils.data.Dataset) of a specific mode and image set.
        
        Args:
        - mode (str): the mode of training to select. See available modes with `available_modes`.
        - remove_invalid (bool): if set to True, deletes any NaN/corrupt items in the image set
                                 pertaining to the relevant mode. In the False case, either NaN
                                 values are substituted with empty values or an error is thrown,
                                 depending on the mode selected.
        - image_set (str, Optional): the image set to pull from. Default: all images.
        - transform (Callable, Optional): the transform operation to apply to the images.
        - target_transform (Callable, Optional): the transform operation on the labels.
        - transforms (tuple, Optional): tuple in the format (transform, target_transform). Default
                                        PyTorch transforms are available in the CVTransforms class.
        - resize (tuple[int, int], Optional): if provided, resize all images to exact configuration.
        - normalize (str, Optional): if provided, normalize bounding box/segmentation coordinates
                                     to a specific configuration. Options: 'zeroone', 'full'
        '''
            
        if transforms: transform, target_transform = transforms
        if not self.cleaned: self.parse()
        assert mode.lower().strip() in self.available_modes, 'Desired mode not available.'
        
        dataframe = self.dataframe[[image_set in item for item in self.dataframe['IMAGE_SET_NAME']]]
        if image_set is None: dataframe = self.dataframe
        if len(dataframe) == 0: raise ValueError(f'Image set {image_set} not available.')
        normalization = None
        if mode == 'classification': 
            dataframe = dataframe[list(CVData._classification_cols)]
            id_mapping = {k: i for i, k in enumerate(self.idx_to_class)}
        elif mode == 'detection': 
            dataframe = dataframe[list(CVData._detection_cols)]
            normalization = self.bbox_scale_option
            id_mapping = {k: i for i, k in enumerate(self.idx_to_bbox_class)}
        elif mode == 'segmentation':
            dataframe = dataframe[list(CVData._segmentation_poly_cols if 'POLYGON' in
                                       dataframe else CVData._segmentation_img_cols)]
            normalization = self.seg_scale_option
            id_mapping = {k: i for i, k in enumerate(self.idx_to_seg_class)}
        elif mode == 'inference':
            dataframe = dataframe[list(CVData._inference_cols)]
            id_mapping = None
        elif mode == 'diffusion':
            dataframe = dataframe[list(CVData._diffusion_cols)]
            id_mapping = None
        if remove_invalid:
            print(f'Removed {len(dataframe[dataframe.isna().any(axis=1)])} NaN entries.')
            dataframe = dataframe.dropna()
        else:
            replace_nan = (lambda x: ([] if isinstance(x, float) and isnan(x) else x))
            if mode == 'detection': cols = ['BBOX_CLASS_ID'] # BOX already accounted for in bbox creation
            elif mode == 'segmentation': cols = ['POLYGON', 'SEG_CLASS_ID'] if 'POLYGON' in dataframe else []
            for col in cols: dataframe[col] = dataframe[col].apply(replace_nan)
            for i, row in dataframe.iterrows():
                for val in row.values:
                    if isinstance(val, float) and isnan(val):
                        error = '[CVData] Found NaN values that will cause errors in row:\n'
                        error += str(dataframe.iloc[i])
                        raise ValueError(error)
        
        if len(dataframe) == 0: raise ValueError('[CVData] After cleanup, this dataset is empty.')
        return CVDataset(dataframe, self.root, mode, id_mapping=id_mapping, transform=transform,
                         target_transform=target_transform, resize=resize, store_dim=store_dim,
                         normalize_to=normalize, normalization=normalization)

    def get_dataloader(
        self,
        mode: str,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 1,
        remove_invalid: bool = True,
        store_dim: bool = False,
        image_set: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[tuple[Callable]] = None,
        resize: Optional[tuple[int, int]] = None,
        normalize: Optional[str] = None
    ) -> DataLoader:
        '''
        Retrieve the PyTorch dataloader (torch.utils.data.DataLoader) for this dataset.
        
        Args:
        - mode (str): the mode of training to select. See available modes with `available_modes`.
        - batch_size (int): the batch size of the image. Default: 4.
        - shuffle (bool): whether to shuffle the data before loading. Default: True.
        - num_workers (int): number of workers for the dataloader. Default: 1.
        - remove_invalid (bool): if set to True, deletes any NaN/corrupt items in the image set
                                 pertaining to the relevant mode. In the False case, either NaN
                                 values are substituted with empty values or an error is thrown,
                                 depending on the mode selected.
        - image_set (str, Optional): the image set to pull from. Default: all images.
        - transform (Callable, Optional): the transform operation to apply to the images.
        - target_transform (Callable, Optional): the transform operation on the labels.
        - transforms (tuple, Optional): tuple in the format (transform, target_transform). Default
                                        PyTorch transforms are available in the CVTransforms class.
        - resize (tuple[int, int], Optional): if provided, resize all images to exact configuration.
        - normalize (str, Optional): if provided, normalize bounding box/segmentation coordinates
                                     to a specific configuration. Options: 'zeroone', 'full'
        '''
        return DataLoader(
            self.get_dataset(
                mode,
                remove_invalid=remove_invalid,
                store_dim=store_dim,
                image_set=image_set,
                transform=transform,
                target_transform=target_transform,
                transforms=transforms,
                resize=resize,
                normalize=normalize),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=_collate(mode, store_dim)
        )
    
    def split_image_set(
        self,
        image_set: Union[str, int],
        *new_sets: tuple[str, float],
        inplace: bool = False,
        seed: Optional[int] = None
    ) -> None:
        '''
        Split the existing image set into new image sets. If inplace is True, the existing image
        set will receive the percentage that is missing from the rest of the sets, or deleted if
        the other sets add up to 1.
        
        Args:
        - image_set (str, int): the old image set name to split. Accepts both name and ID.
        - new_sets (tuple[str, float]): each entry of new_sets has a name for the set accompanied
                                        with a float to represent the percentage to split data into.
        - inplace (bool): whether to perform the operation inplace on the existing image set. If
                          False, then the new sets are required to add up to exactly 100% of the
                          compositions. If True, any remaining percentages less than 100% will be
                          filled back into the old image set. Default: False.
        - seed (int, Optional): the seed to use for the operation, in case consistent dataset
                      manipulation in memory is required. Default: None
        '''
        # checks before splitting
        mode = 'name' if isinstance(image_set, str) else 'id'
        if mode == 'name' and image_set not in self.image_set_to_idx: 
            raise ValueError("Image set doesn't exist!")
        if mode == 'name' and any(new_set[0] in self.image_set_to_idx for new_set in new_sets): 
            raise ValueError(f'New set name already exists!')
        if mode == 'id' and image_set not in self.idx_to_image_set: 
            raise ValueError("Image set ID doesn't exist!")
        if mode == 'id' and any(new_set[0] in self.idx_to_image_set for new_set in new_sets): 
            raise ValueError(f'New set ID already exists!')

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
        if mode == 'name':
            for k in new_sets:
                next_id = next_avail_id(self.idx_to_image_set)
                self.idx_to_image_set[next_id] = k[0]
                self.image_set_to_idx[k[0]] = next_id
        else:
            for k in new_sets:
                self.idx_to_image_set[k[0]] = str(k[0])
                self.image_set_to_idx[str(k[0])] = k[0]

        # assign image sets
        for _, row in self.dataframe.iterrows():
            if image_set not in row['IMAGE_SET_NAME']: continue
            partition = random.random()
            running_sum = 0
            for _, next_set in enumerate(new_sets):
                if running_sum <= partition <= running_sum + next_set[1]:
                    next_id = next_set[0] if mode == 'id' else self.image_set_to_idx[next_set[0]]
                    next_name = str(next_set[0]) if mode == 'id' else next_set[0]
                    if inplace:
                        index = row['IMAGE_SET_NAME'].index(image_set)
                        row['IMAGE_SET_NAME'][index] = next_name
                        row['IMAGE_SET_ID'][index] = next_id
                    else: 
                        row['IMAGE_SET_NAME'].append(next_name)
                        row['IMAGE_SET_ID'].append(next_id)
                    break
                running_sum += next_set[1]
        if inplace: self.clear_image_sets(image_set)
    
    def get_image_set(
        self,
        image_set: Union[str, int]
    ) -> DataFrame:
        '''
        Retrieve the sub-DataFrame which contains all images in a specific image set.
        
        Args:
        - image_set (str, int): the image set. Accepts both string and int.
        '''
        if isinstance(image_set, str):
            return self.dataframe[self.dataframe['IMAGE_SET_NAME'].apply(lambda x: image_set in x)]
        return self.dataframe[self.dataframe['IMAGE_SET_ID'].apply(lambda x: image_set in x)]
    
    def clear_image_sets(
        self,
        sets: Optional[list[Union[str, int]]] = None
    ) -> None:
        '''
        Clear image sets from the dict if they contain no elements.
        
        Args:
        - sets (list[str | int], Optional): If defined, only scan the provided list, otherwise
                                                  scan all sets. Default: None.
        '''
        to_pop = []
        if sets is None: sets = self.image_set_to_idx
        for image_set in sets:
            if len(self.get_image_set(image_set)) == 0:
                to_pop.append(image_set)
        for image_set in to_pop:
            if isinstance(image_set, str):
                index = self.image_set_to_idx.pop(image_set, None)
                if index is not None: self.idx_to_image_set.pop(index, None)
            else:
                self.image_set_to_idx.pop(str(image_set), None)
                self.idx_to_image_set.pop(image_set, None)
    
    def delete_image_set(
        self,
        image_set: Union[str, int]
    ) -> None:
        '''
        Delete image set from all entries. If an entry has only that image set, replace with the
        default dataset.
        
        Args:
        - image_set (str, int): the image set to delete. Accepts both name and ID.
        '''
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
        Save the dataset into CVData json format.
        
        Args:
        - filename (str): the filename to save the dataset.
        - overwrite (bool): whether to overwrite the file if it already exists. Default: False.
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
            'get_img_dim': self.get_img_dim,
            'get_md5_hashes': self.get_md5_hashes,
            'bbox_scale_option': self.bbox_scale_option,
            'seg_scale_option': self.seg_scale_option,
            'available_modes': self.available_modes,
            'cleaned': self.cleaned,
        }
        if os.path.exists(filename) and not overwrite:
            raise ValueError(f'File already exists: {filename}')
        with open(filename, 'w') as f:
            f.write(json.dumps(this))

    @classmethod
    def load(cls, filename: str) -> Self:
        '''
        Load a CVData object from file. Warning: do not load any json files that you did not create.
        This method uses jsonpickle, an insecure loading system with potential for arbitrary Python
        code execution.
        
        Args:
        - filename (str): the filename to load the data from.
        '''
        try:
            print('[CVData] Loading dataset...')
            start = time.time()
            with open(filename, 'r') as f:
                data = json.load(f)
            this: CVData = cls(
                data['root'],
                jsonpickle.decode(data['form'], keys=True),
                remove_invalid=data['remove_invalid'],
                get_img_dim=data['get_img_dim'],
                get_md5_hashes=data['get_md5_hashes'],
                bbox_scale_option=data['bbox_scale_option'],
                seg_scale_option=data['seg_scale_option']
            )
            this.dataframe = DataFrame.from_dict(json.loads(data['dataframe']))
            this.image_set_to_idx = data['image_set_to_idx']
            this.idx_to_image_set = data['idx_to_image_set']
            this.seg_class_to_idx = data['seg_class_to_idx']
            this.idx_to_seg_class = data['idx_to_seg_class']
            this.bbox_class_to_idx = data['bbox_class_to_idx']
            this.idx_to_bbox_class = data['idx_to_bbox_class']
            this.available_modes = data['available_modes']
            this.cleaned = data['cleaned']
            end = time.time()
            print(f'[CVData] Loaded dataset! ({end - start}s)')
        except Exception:
            print('[CVData] This is not a CVData dataset!')
            return None
        return this
        
    def sample_image(
        self,
        dpi: float = 1200,
        mode: Optional[str] = None,
        idx: Optional[int] = None
    ) -> None:
        if not self.cleaned: raise ValueError('Run parse() to populate data first!')
        if mode is not None: 
            mode = union(mode)
            for try_mode in mode:
                if try_mode not in self.available_modes: 
                    raise ValueError(f'Mode {try_mode} not available.')
        else: mode = self.available_modes
        item = self.dataframe.iloc[idx] if idx is not None else self.dataframe.sample().iloc[0]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.uint8)
        ])
        image = transform(open_image(item['ABSOLUTE_FILE']).convert('RGB'))
        plt.figure(dpi=dpi)
        if 'classification' in mode:
            print(f'[CVData] Image Class ID/Name: {item["CLASS_ID"]}/{item["CLASS_NAME"]}')
        if 'detection' in mode:
            if len(item['BOX']) != 0:
                image = draw_bounding_boxes(image,
                                            torch.stack([FloatTensor(box) for box in item['BOX']]),
                                            width=3,
                                            labels=item['BBOX_CLASS_NAME'],
                                            colors='red')
            else: print('[CVData] Warning: Image has no bounding boxes.')
        if 'segmentation' in mode:
            _, axarr = plt.subplots(ncols=2)
            axarr[0].imshow(image.permute(1, 2, 0))
            if 'ABSOLUTE_FILE_SEG' in item: 
                mask = F.to_tensor(open_image(item['ABSOLUTE_FILE_SEG']))
                axarr[1].imshow(mask.permute(1, 2, 0))
            else:
                assert len(item['POLYGON']) == len(item['SEG_CLASS_ID']), \
                    'SEG_CLASS_ID and POLYGON len mismatch'
                mask = asarray(imread(item['ABSOLUTE_FILE'], IMREAD_GRAYSCALE), dtype=int32)
                mask = full_like(mask, next_avail_id(self.idx_to_seg_class))
                for class_id, polygon in zip(item['SEG_CLASS_ID'], item['POLYGON']):
                    mask = fillPoly(mask, pts=[asarray(polygon, dtype=int32)],
                                    color=class_id)
                mask = torch.from_numpy(asarray(mask))
                axarr[1].imshow(mask)
        else:
            plt.imshow(image.permute(1, 2, 0))

    def inference(
        self,
        image: torch.Tensor,
        label: Any,
        result: Any,
        dpi: float = 1200,
        mode: str = None
    ) -> None:
        if not self.cleaned: raise ValueError('Run parse() to populate data first!')
        if mode is None: raise ValueError('Must specify a valid mode.')
        if mode not in self.available_modes: 
            raise ValueError(f'Mode {mode} not available.')
        plt.figure(dpi=dpi)
        if mode == 'classification':
            _, pred = torch.max(result, dim=1)
            print(f'[CVData] Image Class ID/Name: {pred}/{self.idx_to_class[pred]}')
        elif mode == 'detection':
            boxes = result['boxes']
            labels = result['labels']
            if len(boxes) != 0:
                image = draw_bounding_boxes(image, boxes, width=3, colors='red',
                                            labels=[self.idx_to_bbox_class[label] for label in labels])
            else: print('[CVData] Warning: Image has no bounding boxes.')
        if mode == 'segmentation':
            _, axarr = plt.subplots(ncols=2)
            axarr[0].imshow(image.permute(1, 2, 0))
            mask = result
            raise NotImplementedError("I haven't implemented this yet. Please ping me so we can figure it out")
            axarr[1].imshow(mask.permute(1, 2, 0))
        else:
            plt.imshow(image.permute(1, 2, 0))

class CVDataset(VisionDataset):
    '''
    Dataset implementation for the CVData environment.
    
    Args:
    - df (DataFrame): the dataframe from CVData.
    - root (str): the root of the dataset folder.
    - mode (str): the mode of the data to retrieve, i.e. classification, segmentation, detection.
    - id_mapping (dict[int, int]): the id mapping from the dataframe to retrieve class names.
                                   this is used primarily as a safety feature in order to make sure
                                   that used IDs are provided in order starting from 0 without holes
                                   so that training works properly.
    - image_type (str): the type of the image to export, to convert PIL images to. Default: 'RGB'.
                        Also accepts 'L' and 'CMYK'.
    - normalization (str): the type of normalization that the dataset currently is formatted in,
                           for box and polygon items. Accepts 'full' or 'zeroone'.
    - normalize_to (str): the type of normalization that the dataset is to be resized to, for box
                          and polygon items. Accepts 'full' or 'zeroone'.
    - transform (Callable, Optional): the transform operation to apply to the images.
    - target_transform (Callable, Optional): the transform operation on the labels.
    '''
    def __init__(
        self,
        df: DataFrame,
        root: str,
        mode: str,
        id_mapping: Optional[dict[int, int]],
        image_type: str = 'RGB',
        normalization: str = 'full',
        store_dim: bool = False,
        resize: Optional[tuple[int, int]] = None,
        normalize_to: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.dataframe = df
        self.data = self.dataframe.to_dict('records')
        self.mode = mode
        self.image_type = image_type
        self.id_mapping = id_mapping
        self.store_dim = store_dim
        self.resize = resize
        self.normalize_to = normalize_to
        self.normalization = normalization
        print(f'Resize: {self.resize}')
        if self.mode == 'segmentation': self.default = len(self.id_mapping)
        super().__init__(root, transforms=None, transform=transform, target_transform=target_transform)

    def __len__(self):
        return len(self.dataframe)

    def _get_class_labels(self, item: Series) -> Tensor:
        return self.id_mapping[int(item['CLASS_ID'])]

    def _get_bbox_labels(self, item: Series) -> dict[str, Tensor]:
        # execute checks
        assert len(item['BOX']) == len(item['BBOX_CLASS_ID']), \
            'BOX and BBOX_CLASS_ID len mismatch'
        class_ids = list(map(lambda x: self.id_mapping[x], item['BBOX_CLASS_ID']))
        boxes = item['BOX']
        if self.resize: factor_resize = self.resize
        else: factor_resize = (1, 1)
        if self.normalization == 'full':
            factor_norm = item['IMAGE_DIM']
        else:
            factor_norm = (1, 1)
        apply_resize = lambda p: (p[0] * factor_resize[0] / factor_norm[0],
                                  p[1] * factor_resize[1] / factor_norm[1],
                                  p[2] * factor_resize[0] / factor_norm[0],
                                  p[3] * factor_resize[1] / factor_norm[1])
        bbox_tensors = [FloatTensor(apply_resize(box)) for box in boxes]
        if self.store_dim:
            return {'boxes': torch.stack(bbox_tensors), 'labels': LongTensor(class_ids), 'dim': item['IMAGE_DIM']}
        else:
            return {'boxes': torch.stack(bbox_tensors), 'labels': LongTensor(class_ids)}

    def _get_seg_labels(self, item: Series) -> Tensor:
        if 'ABSOLUTE_FILE_SEG' in item:
            mask = F.to_tensor(open_image(item['ABSOLUTE_FILE_SEG']))
            if self.resize:
                return F.resize(mask, [self.resize[1], self.resize[0]])
            return mask
        assert len(item['POLYGON']) == len(item['SEG_CLASS_ID']), \
            'SEG_CLASS_ID and POLYGON len mismatch'
        if self.resize is not None:
            mask = full(self.resize, self.default, dtype=int32)
            factor_resize = self.resize
        else:
            mask = full(item['IMAGE_DIM'], self.default, dtype=int32)
            factor_resize = (1, 1)
        if self.normalization == 'full': factor_norm = item['IMAGE_DIM']
        else: factor_norm = (1, 1)
        apply_resize = lambda p: (p[0] * factor_resize[0] / factor_norm[0],
                                  p[1] * factor_resize[1] / factor_norm[1])
        for class_id, polygon in zip(item['SEG_CLASS_ID'], item['POLYGON']):
            if self.resize is not None: polygon = list(map(apply_resize, polygon))
            mask = fillPoly(mask, pts=[asarray(polygon, dtype=int32)],
                            color=self.id_mapping[class_id])
        mask = torch.from_numpy(asarray(mask)).unsqueeze(-1).permute(2, 0, 1)
        return mask

    def __getitem__(self, idx):
        item: dict = self.data[idx]
        image: Tensor = F.to_tensor(open_image(item.get('ABSOLUTE_FILE')).convert(self.image_type))
        if self.resize: image = F.resize(image, [self.resize[1], self.resize[0]])
        label: dict[str, Tensor]
        if self.mode == 'inference' or self.mode == 'diffusion':
            if self.transform:
                image = self.transform(image)
            return image, {'dim': item['IMAGE_DIM']}
        if self.mode == 'classification':
            label = self._get_class_labels(item)
        elif self.mode == 'detection':
            label = self._get_bbox_labels(item)
        elif self.mode == 'segmentation':
            label = self._get_seg_labels(item)
        if self.transforms:
            image, label = self.transforms(image, label)
        if self.store_dim:
            return image, {'label': label, 'dim': item['IMAGE_DIM']}
        return image, label
