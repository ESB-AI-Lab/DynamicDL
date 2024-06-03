import os
import time
import json
from hashlib import md5
from functools import partial
import random
from typing import Union, Optional, Callable, Iterable, Tuple

import cv2
from tqdm import tqdm
import numpy as np
import jsonpickle
from pandas import DataFrame
from pandas import isna
from pandas.core.series import Series
from torch.utils.data import DataLoader
from torch import FloatTensor
import torch
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
from torchvision import transforms as T
from PIL.Image import open as open_image
import matplotlib.pyplot as plt
from typing_extensions import Self

from ._utils import next_avail_id, union, config
from ._warnings import Warnings
from ._main._engine import populate_data
from ._main._transforms import Transforms
from ._main._collate import _collate
from .dynamicds import DynamicDS

class DynamicData:
    '''
    Main dataset class. Accepts root directory path and dictionary form of the structure.
    DynamicDL expands a generic dataset form and interprets it through a series of recursive
    hierarchical inheritances, to flatten the dataset into a list of entries fit for image
    processing.
    
    :param root: The root directory to access the dataset.
    :type root: str
    :param form: The form of the dataset. See documentation for further details on valid forms.
    :type form: dict
    :param bbox_scale_option: Choose from either `auto`, `zeroone`, or `full` scale options to
        define, or leave empty for automatic. `zeroone` assumes detection coordinates to be
        interpreted on a 0-1 scale as ratios dependent on image size. `full` leaves coordinates
        as is. `auto` for auto-detection. Default: `auto`
    :type bbox_scale_option: str
    :param seg_scale_option: Choose from either `auto`, `zeroone`, or `full` scale options to
        define, or leave empty for automatic. `zeroone` assumes segmentation coordinates to be
        interpreted on a 0-1 scale as ratios dependent on image size. `full` leaves coordinates
        as is. `auto` for auto-detection. Default: `auto`
    :type seg_scale_option: str
    :param get_md5_hashes: When set to True, create a new column which finds md5 hashes for each
        image available, and makes sure there are no duplicates. Default: `False`
    :type get_md5_hashes: bool
    :param purge_duplicates: When set to True, remove all duplicate image entries. Duplicate images
        are defined by having the same md5 hash, so this has no effect when `get_md5_hashes` is 
        `False`. When set to `False`, do not purge duplicates. Default: `None`
    :type purge_duplicates: Optional[bool]
    '''

    _modes: dict[str, set[str]] = config['MODES']

    _BBOX_MODES = config['BBOX_MODES']
    _BBOX_COLS = config['BBOX_COLS']

    _scale_options = ('zeroone', 'full')

    def __init__(
        self,
        root: str,
        form: dict,
        bbox_scale_option: str = 'auto',
        seg_scale_option: str = 'auto',
        get_md5_hashes: bool = False,
        purge_duplicates: Optional[bool] = None
    ) -> None:
        self.root = root
        self.form = form
        self.image_set_to_idx = {}
        self.idx_to_image_set = {}
        self.seg_class_to_idx = {}
        self.idx_to_seg_class = {}
        self.bbox_class_to_idx = {}
        self.idx_to_bbox_class = {}
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.available_modes = []
        self.cleaned = False
        self.get_md5_hashes = get_md5_hashes
        self.purge_duplicates = purge_duplicates
        self.bbox_scale_option = bbox_scale_option
        self.seg_scale_option = seg_scale_option

    def parse(self, override: bool = False, verbose: bool = False) -> None:
        '''
        Must be called to instantiate the data in the dataset instance. Performs the recursive
        populate_data algorithm and creates the dataframe, and then cleans up the data.
        
        :param override: Whether to overwrite existing data if it has already been parsed and
            cleaned. Default: `False`
        :type override: bool
        :param verbose: Whether to show more details about the merging process. May have an impact
            on runtime. Default: `False`
        :type verbose: bool
        '''
        print('[DynamicData] Parsing data...')
        start = time.time()
        if self.cleaned and not override:
            Warnings.error('already_parsed')
        data = populate_data(self.root, self.form, verbose=verbose)
        entries = [{key: val.value for key, val in item.data.items()} for item in data]
        self.dataframe = DataFrame(entries)
        end = time.time()
        print(f'[DynamicData] Parsed! ({(end - start):.3f}s)')
        start = time.time()
        self._cleanup(verbose=verbose)
        end = time.time()
        print(f'[DynamicData] Cleaned! ({(end - start):.3f}s)')
        print(self._get_statistics())

    def _cleanup(self, verbose: bool = False) -> None:
        '''
        Run cleanup and sanity checks on all data. Assigns IDs to name-only values.
        '''
        print('[DynamicData] Cleaning up data...')

        if 'ABSOLUTE_FILE' not in self.dataframe:
            Warnings.error('no_images')

        # sort by image id first to prevent randomness
        if 'IMAGE_ID' in self.dataframe:
            self.dataframe.sort_values('IMAGE_ID', ignore_index=True, inplace=True)
        else:
            self.dataframe.sort_values('IMAGE_NAME', ignore_index=True, inplace=True)
            self.dataframe['IMAGE_ID'] = self.dataframe.index

        # get image sizes
        self._get_img_sizes()

        # get md5 hashes
        if self.get_md5_hashes:
            self._get_md5_hashes()

        # convert bounding boxes into proper format and store under 'BOX'
        self._convert_bbox()

        if 'BOX' in self.dataframe:
            self._get_box_scale()
            self._convert_box_scale()

        if 'POLYGON' in self.dataframe:
            self._get_seg_scale()
            self._convert_seg_scale()

        # assign ids
        self._cleanup_id()
        self._process_ids('CLASS', redundant=False, verbose=verbose)
        self._process_ids('SEG_CLASS', redundant=True, verbose=verbose)
        self._process_ids('BBOX_CLASS', redundant=True, verbose=verbose)

        # check available columns to determine mode availability
        self.available_modes = DynamicData._get_modes(self.dataframe)

        # cleanup image sets
        self._cleanup_image_sets()
        self.cleaned = True

    @staticmethod
    def _get_modes(df: DataFrame) -> list:
        modes = [mode for mode, subset in DynamicData._modes.items() if subset.issubset(df.columns)]
        return modes

    def _get_statistics(self):
        data = '[DynamicData] Dataset statistics:\n'
        data += f'       | Available modes: {", ".join(self.available_modes)}\n'
        data += f'       | Images: {len(self.dataframe)}\n'
        for mode in self.available_modes:
            count = len(self.dataframe) - len(self.dataframe[
                self.dataframe[list(DynamicData._modes[mode])].isna().any(axis=1)
            ])
            data += f'       | Complete entries for {mode}: {count}\n'

        if 'detection' in self.available_modes:
            data += f'       | Bounding box scaling option: {self.bbox_scale_option}\n'
        if ('segmentation_poly' in self.available_modes or 
            'segmentation_mask' in self.available_modes):
            data += f'       | Segmentation object scaling option: {self.seg_scale_option}\n'
        return data.strip()

    def _process_ids(self, name: str, redundant: bool = False, verbose: bool = False) -> None:
        if f'{name}_NAME' in self.dataframe:
            if f'{name}_ID' not in self.dataframe:
                call = partial(self._assign_ids, redundant=redundant)
            else: call = partial(self._validate_ids, redundant=redundant)
            result = call(f'{name}')
            setattr(self, f'{name.lower()}_to_idx', result[0])
            setattr(self, f'idx_to_{name.lower()}', result[1])
        elif f'{name}_ID' in self.dataframe:
            setattr(self, f'idx_to_{name.lower()}',
                    {i: str(i) for item in self.dataframe[f'{name}_ID']
                     if isinstance(item, list) for i in item})
            setattr(self, f'{name.lower()}_to_idx',
                    {str(i): i for item in self.dataframe[f'{name}_ID']
                     if isinstance(item, list) for i in item})
            names = [list(map(lambda x: getattr(self, f'idx_to_{name.lower()}')[x], i))
                     if isinstance(i, list) else [] for i in self.dataframe[f'{name}_ID']]
            self.dataframe[f'{name}_NAME'] = names
        else:
            return
        self._patch_ids(
            name,
            getattr(self, f'{name.lower()}_to_idx'),
            getattr(self, f'idx_to_{name.lower()}'),
            redundant=redundant,
            verbose=verbose
        )

    def _get_img_sizes(self) -> None:
        self.dataframe['IMAGE_DIM'] = [open_image(filename).size if isinstance(filename, str)
                                       else np.nan for filename in self.dataframe['ABSOLUTE_FILE']]

    def _get_md5_hashes(self) -> None:
        hashes = [md5(open_image(item).tobytes()).hexdigest() for item in
                  tqdm(self.dataframe['ABSOLUTE_FILE'],desc='[DynamicData] Calculating md5 hashes')]
        counter = {}
        for i, md5hash in enumerate(hashes):
            counter[md5hash] = counter.get(md5hash, []) + [i]
        duplicates = [locs for locs in counter.values() if len(locs) > 1]
        self.dataframe['MD5'] = hashes
        if duplicates:
            if self.purge_duplicates is None:
                strmsg = ''
                for i, locs in enumerate(duplicates):
                    locstr = ", ".join([self.dataframe["IMAGE_NAME"].iloc[loc] for loc in locs])
                    strmsg += f'\n{i}: {locstr}'
                Warnings.error('duplicate_images', duplicates=duplicates)
            if self.purge_duplicates:
                dupes = []
                for locs in duplicates:
                    dupes += locs[1:]
                self.dataframe.drop(dupes, inplace=True)

    def _get_box_scale(self) -> None:
        if self.bbox_scale_option == 'auto':
            for i, boxes in enumerate(self.dataframe['BOX']):
                if any(coord > 1 for box in boxes for coord in box):
                    self.bbox_scale_option = 'full'
                if any(coord < 0 for box in boxes for coord in box):
                    Warnings.error('invalid_scale_data_bbox', id=i)
            if self.bbox_scale_option == 'full':
                print('[DynamicData] Detected full size bounding box scale option')
                return
            print('[DynamicData] Detected [0, 1] bounding box scale option to be converted to full '
                  'size')
            self.bbox_scale_option = 'zeroone'
        if self.bbox_scale_option not in DynamicData._scale_options:
            Warnings.error('invalid_scale', scale=self.bbox_scale_option)

    def _get_seg_scale(self) -> None:
        if self.seg_scale_option == 'auto':
            for i, shapes in enumerate(self.dataframe['POLYGON']):
                if any(val > 1 for shape in shapes for coord in shape for val in coord):
                    self.seg_scale_option = 'full'
                    print('[DynamicData] Detected full size segmentation scale option')
                    return
                if any(coord < 0 for shape in shapes for coord in shape):
                    Warnings.error('invalid_scale_data', id=i)
            print('[DynamicData] Detected [0, 1] segmentation scale option to be converted to full '
                  'size')
            self.seg_scale_option = 'zeroone'
        if self.seg_scale_option not in DynamicData._scale_options:
            Warnings.error('invalid_scale', scale=self.seg_scale_option)

    def _convert_box_scale(self) -> None:
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
            if isna(i) or (isinstance(v, float) and isna(v)):
                return
            i = int(i)
            if v in name_to_idx and name_to_idx[v] != i:
                Warnings.error(
                    'invalid_id_map',
                    type=name,
                    i=i,
                    v=v,
                    expect=name_to_idx[v]
                )
            else:
                name_to_idx[v] = i

        name_to_idx = {}
        for i, (ids, vals) in self.dataframe[[f'{name}_ID', f'{name}_NAME']].iterrows():
            if (isinstance(ids, float) and isna(ids)) or (isinstance(vals, float) and isna(vals)):
                continue
            if redundant:
                if len(ids) != len(vals):
                    Warnings.error(
                        'row_mismatch',
                        name1=f'{name}_ID',
                        name2=f'{name}_NAME',
                        len1=len(ids),
                        len2=len(vals)
                    )
                for i, v in zip(ids, vals):
                    check(i, v, name_to_idx)
            else:
                check(ids, vals, name_to_idx)
        return name_to_idx, {v: k for k, v in name_to_idx.items()}

    def _patch_ids(
        self,
        name: str,
        name_to_idx: dict,
        idx_to_name: dict,
        redundant: bool = False,
        verbose: bool = False
    ) -> None:
        '''Patch nan values of ids/vals accordingly.'''
        ctr = 0
        if not redundant:
            for i, (ids, vals) in self.dataframe[[f'{name}_ID', f'{name}_NAME']].iterrows():
                if isna(ids) and isinstance(vals, float) and isna(vals):
                    ctr += 1
                    if verbose:
                        print(f'Found missing {name} id/name at row {i}')
                    continue
                if isna(ids):
                    self.dataframe.at[i, f'{name}_ID'] = name_to_idx[vals]
                if isinstance(vals, float) and isna(vals):
                    self.dataframe.at[i, f'{name}_NAME'] = idx_to_name[ids]
            if ctr:
                print(f'[DynamicData] Patched {ctr} id/name pairs for {name}.')
                if not verbose:
                    print('[DynamicData] Use parse() with verbose=True to see all invalid entries.')
            return
        id_vals = []
        name_vals = []
        for i, (ids, vals) in self.dataframe[[f'{name}_ID', f'{name}_NAME']].iterrows():
            if isinstance(ids, float) and isinstance(vals, float):
                ctr += 1
                if verbose:
                    print(f'Found missing {name} id/name at row {i}')
                id_vals.append([])
                name_vals.append([])
                continue
            if isinstance(ids, float):
                id_vals.append(list(map(lambda x: name_to_idx[x], vals)))
            else:
                id_vals.append(ids)
            if isinstance(vals, float):
                name_vals.append(list(map(lambda x: idx_to_name[x], ids)))
            else:
                name_vals.append(vals)
        self.dataframe[f'{name}_ID'] = id_vals
        self.dataframe[f'{name}_NAME'] = name_vals
        if ctr:
            print(f'[DynamicData] Patched {ctr} id/name pairs for {name}.')
            if not verbose:
                print('[DynamicData] Use parse() with verbose=True to see all invalid entries.')

    def _assign_ids(self, name: str, default=False, redundant=False) -> \
            tuple[dict[str, int], dict[int, str]]:
        sets = set()
        default_value = ['default'] if redundant else 'default'
        if default:
            self.dataframe.loc[self.dataframe[f'{name}_NAME'].isna(), f'{name}_NAME'] = \
                self.dataframe.loc[self.dataframe[f'{name}_NAME'].isna(), f'{name}_NAME'].apply(
                    lambda x: default_value)
        for v in self.dataframe[f'{name}_NAME']:
            if isinstance(v, float):
                continue
            if redundant:
                sets.update(v)
            else:
                sets.add(v)
        name_to_idx = {v: i for i, v in enumerate(sets)}
        idx_to_name = {v: k for k, v in name_to_idx.items()}
        if redundant:
            self.dataframe[f'{name}_ID'] = self.dataframe[f'{name}_NAME'].apply(lambda x:
                np.nan if isinstance(x, float) else list(map(lambda y: name_to_idx[y], x)))
        else:
            self.dataframe[f'{name}_ID'] = self.dataframe[f'{name}_NAME'].apply(lambda x:
                np.nan if isinstance(x, float) else name_to_idx[x])
        return name_to_idx, idx_to_name

    def _convert_bbox(self) -> None:
        cols, funcs = None, None
        for colset, key_cols, key_funcs in DynamicData._BBOX_MODES:
            if colset.issubset(self.dataframe.columns):
                cols, funcs = key_cols, key_funcs
        if cols is None or funcs is None:
            if any(col in self.dataframe for col in DynamicData._BBOX_COLS):
                Warnings.error(
                    'incomplete_bbox',
                    columns=DynamicData._BBOX_COLS.intersection(self.dataframe.columns)
                )
            return

        def execute_checks(row: Series, cols: tuple):
            if any(isinstance(row[cols[i]], float) for i in range(4)):
                return False
            for x in cols:
                if len(row[x]) != len(row[cols[0]]):
                    Warnings.error(
                        'row_mismatch',
                        name1=cols[0],
                        name2=x,
                        len1=len(row[x]),
                        len2=len(row[cols[0]])
                    )
            return True

        boxes = []
        for i, row in self.dataframe.iterrows():
            if not execute_checks(row, cols):
                boxes.append([])
            else:
                box = []
                for x1, y1, x2, y2 in zip(*[row[cols[i]] for i in range(4)]):
                    box.append((funcs[0]((x1, x2)), funcs[1]((y1, y2)),
                                funcs[2]((x1, x2)), funcs[3]((y1, y2))))
                boxes.append(box)
        self.dataframe['BOX'] = boxes
        self.dataframe.drop(DynamicData._BBOX_COLS.difference({'BBOX_CLASS_ID', 'BBOX_CLASS_NAME'}),
                            axis=1, inplace=True, errors='ignore')

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

    def _cleanup_id(self) -> None:
        cols = ['CLASS_ID', 'IMAGE_ID']
        for col in cols:
            if col not in self.dataframe:
                continue
            self.dataframe[col] = self.dataframe[col].astype('Int64')

    def get_transforms(
        self,
        mode: str = 'inference',
        calculate_stats: bool = True,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        normalize: bool = True,
        remove_invalid: bool = True,
        resize: Optional[Tuple[int, ...]] = None
    ) -> Tuple[Optional[Callable], ...]:
        '''
        Retrieve the default standard image/label transforms for specified mode.
        
        :param mode: Choose a mode out of available modes for the transforms. Each mode has
            slightly altered standard transforms.
        :type mode: str
        :param calculate_stats: When set to True, calculate the statistics for the entire
            dataset, overriding default mean/std kwarg (defaults from ImageNet). Use this feature
            when dataset mean/std is unknown and differs significantly from ImageNet. Default: True.
        :type calculate_stats: bool
        :param mean: Default mean statistics for the dataset. Has no effect when
            `calculate_stats = True`. Default: ImageNet values.
        :type mean: Tuple[float, ...]
        :param std: Default mean statistics for the dataset. Has no effect when
            `calculate_stats = True`. Default: ImageNet values.
        :type std: Tuple[float, ...]
        :param normalize: When set to True, normalize the dataset according to some
            mean/std values, either from calculated stats or ImageNet default. This statement is
            overriden when `calculate_stats` is set to True. Default: True.
        :type normalize: bool
        :param remove_invalid: Remove invalid entries when calculating the statistics, assuming
            `calculate_stats` is set to `True`. If `calculate_stats` is `False`, this value has no
            effect. Default: `True`.
        :type remove_invalid: bool
        :param resize: Resize to specific tuple dimensions before calculating statistics, only when
            `calculate_stats` is set to True, just like `remove_invalid`. Default: `None`.
        :type resize: Optional[Tuple[int, ...]]
        :return: A tuple of two callable transforms, the first being the standard image transform
            and the latter being the standard target transform.
        :rtype: Tuple[Optional[Callable], ...]
        '''
        if not calculate_stats:
            return Transforms.get(mode, resize=resize, normalize=normalize, mean=mean, std=std)
        if mode not in self.available_modes or mode in ('inference', 'diffusion'):
            return (None, None)
        loader = self.get_dataloader(
            mode,
            remove_invalid=remove_invalid,
            image_set=None,
            preset_transform=False,
            transforms=Transforms.get(mode, resize=resize, normalize=False),
            resize=resize,
            batch_size=10,
            num_workers=0,
            shuffle=False
        )

        mean = 0.
        std = 0.
        for images, _ in tqdm(loader, desc='Calculating stats'):
            if isinstance(images, Iterable):
                for image in images:
                    image = image.view(3, -1)
                    mean += image.mean(1)
                    std += image.std(1)
            else:
                batch_samples = images.size(0)
                images = images.view(batch_samples, images.size(1), -1)
                mean += images.mean(2).sum(0)
                std += images.std(2).sum(0)

        mean /= len(loader.dataset)
        std /= len(loader.dataset)

        print(f"[DynamicData] Got mean {mean} and std {std}")
        return Transforms.get(mode, resize=resize, mean=mean, std=std, normalize=True)

    def get_dataset(
        self,
        mode: str = 'inference',
        remove_invalid: bool = True,
        store_dim: bool = False,
        preset_transform: bool = True,
        calculate_stats: bool = True,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        normalize: bool = True,
        image_set: Optional[Union[int, str]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[tuple[Callable]] = None,
        resize: Optional[tuple[int, int]] = None,
        normalize_to: Optional[str] = None
    ) -> 'DynamicDS':
        '''
        Retrieve the PyTorch dataset (`torch.utils.data.Dataset`) of a specific mode and image set.
        
        :param mode: The mode of training to select. See available modes with `available_modes`.
        :type mode: str
        :param remove_invalid: If set to True, deletes any NaN/corrupt items in the image set
            pertaining to the relevant mode. In the False case, either NaN values are substituted
            with empty values or an error is thrown, depending on the mode selected.
        :type remove_invalid: bool
        :param store_dim: If set to True, the labels in the dataset will return a dict with two
            keys. `label` contains the standard PyTorch labels and `dim` contains the image's
            former dimensions.
        :type store_dim: bool
        :param preset_transform: Whether to use default preset transforms. Consists of normalization
            with either calculated mean of the dataset about to be used or standard ImageNet
            statistics depending on `calculate_stats`. Default: `True`
        :type preset_transform: bool
        :param calculate_stats: Whether to calculate mean and std for this dataset to be used in
            normalization transforms. If False, uses ImageNet default weights. Only has effect
            when `preset_transform` is set to `True`. Default: `True`
        :type calculate_stats: bool
        :param mean: Default mean statistics for the dataset. Has no effect when
            `calculate_stats = True`. Default: ImageNet values.
        :type mean: Tuple[float, ...]
        :param std: Default mean statistics for the dataset. Has no effect when
            `calculate_stats = True`. Default: ImageNet values.
        :type std: Tuple[float, ...]
        :param normalize: When set to `True`, normalize the dataset according to some mean/std
            values, either from calculated stats or ImageNet default. This statement is overriden
            when `calculate_stats` is set to `True`. Default: `True`.
        :type normalize: bool
        :param image_set: The image set to pull from. Default: all images.
        :type image_set: Optional[str]
        :param transform: The transform operation to apply to the images.
        :type transform: Optional[Callable]
        :param target_transform: The transform operation to apply to the labels.
        :type target_transform: Optional[Callable]
        :param transforms: Tuple in the format `(transform, target_transform)`. Obtain default
            transforms from `DynamicData.get_transforms()`, or supply your own.
        :type transforms: Optional[Tuple[Optional[Callable], ...]]
        :param resize: If provided, resize all images to exact `(width, height)` configuration.
        :type resize: Optional[Tuple[int, ...]]
        :param normalize_to: If provided, normalize bounding box/segmentation coordinates to a
            specific configuration. Options: 'zeroone', 'full'
        :type normalize_to: Optional[str]
        '''
        if not self.cleaned:
            self.parse()
        if mode.lower().strip() not in self.available_modes:
            Warnings.error('mode_unavailable', mode=mode.lower().strip())

        if transforms:
            transform, target_transform = transforms
        elif preset_transform:
            transform, target_transform = self.get_transforms(
                mode=mode,
                remove_invalid=remove_invalid,
                resize=resize,
                calculate_stats=calculate_stats,
                mean=mean,
                std=std,
                normalize=normalize
            )

        imgset_mode = 'name' if isinstance(image_set, str) else 'id'
        dataframe = self.dataframe[[image_set in item for item in
                                    self.dataframe[f'IMAGE_SET_{imgset_mode.upper()}']]]
        if image_set is None:
            dataframe = self.dataframe
        if len(dataframe) == 0:
            Warnings.error('image_set_missing', imgset_name=imgset_mode, image_set=image_set)
        normalization = None
        dataframe = dataframe[list(DynamicData._modes[mode])]
        if mode == 'classification':
            id_mapping = {k: i for i, k in enumerate(self.idx_to_class)}
        elif mode == 'detection':
            normalization = self.bbox_scale_option
            id_mapping = {k: i for i, k in enumerate(self.idx_to_bbox_class)}
        elif mode == 'segmentation_mask' or mode == 'segmentation_poly':
            normalization = self.seg_scale_option
            id_mapping = {k: i for i, k in enumerate(self.idx_to_seg_class)}
        elif mode == 'inference' or mode == 'diffusion':
            id_mapping = None
        if remove_invalid:
            dataframe = dataframe.dropna()
            if mode == 'detection':
                start = len(dataframe)
                dataframe = dataframe[dataframe['BOX'].apply(lambda x: len(x) != 0)]
                end = len(dataframe)
                print(f'[DynamicData] Removed {start - end} empty entries from data.')
        else:
            replace_nan = (lambda x: ([] if isinstance(x, float) and isna(x) else x))
            cols = []
            if mode == 'detection':
                cols = ['BBOX_CLASS_ID'] # BOX already accounted for in bbox creation
            elif mode == 'segmentation_poly':
                cols = ['POLYGON', 'SEG_CLASS_ID']
            for col in cols:
                dataframe[col] = dataframe[col].apply(replace_nan)
            for i, row in dataframe.iterrows():
                for val in row.values:
                    if isinstance(val, float) and isna(val):
                        row = str(dataframe.iloc[i])
                        Warnings.error('nan_exists', row=row)

        if len(dataframe) == 0:
            Warnings.error('image_set_empty', image_set=image_set)
        return DynamicDS(
            dataframe,
            self.root,
            mode,
            id_mapping=id_mapping,
            transform=transform,
            target_transform=target_transform,
            resize=resize,
            store_dim=store_dim,
            normalize_to=normalize_to,
            normalization=normalization
        )

    def get_dataloader(
        self,
        mode: str = 'inference',
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 0,
        remove_invalid: bool = True,
        store_dim: bool = False,
        preset_transform: bool = True,
        calculate_stats: bool = True,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        normalize: bool = True,
        image_set: Optional[Union[int, str]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[tuple[Callable]] = None,
        resize: Optional[tuple[int, int]] = None,
        normalize_to: Optional[str] = None
    ) -> DataLoader:
        '''
        Retrieve the PyTorch dataloader (torch.utils.data.DataLoader) for this dataset.
            
        :param mode: The mode of training to select. See available modes with `available_modes`.
        :type mode: str
        :param batch_size: The batch size of the image. Default: 16.
        :type batch_size: int
        :param shuffle: Whether to shuffle the data before loading. Default: `True`.
        :type shuffle: bool
        :param num_workers: Number of workers for the dataloader. Default: 0.
        :type num_workers: int
        :param remove_invalid: If set to True, deletes any NaN/corrupt items in the image set
            pertaining to the relevant mode. In the False case, either NaN values are substituted
            with empty values or an error is thrown, depending on the mode selected.
        :type remove_invalid: bool
        :param store_dim: If set to True, the labels in the dataset will return a dict with two
            keys. `label` contains the standard PyTorch labels and `dim` contains the image's
            former dimensions.
        :type store_dim: bool
        :param preset_transform: Whether to use default preset transforms. Consists of normalization
            with either calculated mean of the dataset about to be used or standard ImageNet
            statistics depending on `calculate_stats`. Default: `True`
        :type preset_transform: bool
        :param calculate_stats: Whether to calculate mean and std for this dataset to be used in
            normalization transforms. If False, uses ImageNet default weights. Only has effect
            when `preset_transform` is set to `True`. Default: `True`
        :type calculate_stats: bool
        :param mean: Default mean statistics for the dataset. Has no effect when
            `calculate_stats = True`. Default: ImageNet values.
        :type mean: Tuple[float, ...]
        :param std: Default mean statistics for the dataset. Has no effect when
            `calculate_stats = True`. Default: ImageNet values.
        :type std: Tuple[float, ...]
        :param normalize: When set to True, normalize the dataset according to some mean/std values,
            either from calculated stats or ImageNet default. This statement is overriden when
            `calculate_stats` is set to T`rue. Default: `True`.
        :type normalize: bool
        :param image_set: The image set to pull from. Default: all images.
        :type image_set: Optional[str]
        :param transform: The transform operation to apply to the images.
        :type transform: Optional[Callable]
        :param target_transform: The transform operation to apply to the labels.
        :type target_transform: Optional[Callable]
        :param transforms: Tuple in the format `(transform, target_transform)`. Obtain default
            transforms from `DynamicData.get_transforms()`, or supply your own.
        :type transforms: Optional[Tuple[Optional[Callable], ...]]
        :param resize: If provided, resize all images to exact `(width, height)` configuration.
        :type resize: Optional[Tuple[int, ...]]
        :param normalize_to: If provided, normalize bounding box/segmentation coordinates to a
            specific configuration. Options: 'zeroone', 'full'
        :type normalize_to: Optional[str]
        '''
        return DataLoader(
            self.get_dataset(
                mode,
                remove_invalid=remove_invalid,
                store_dim=store_dim,
                image_set=image_set,
                preset_transform=preset_transform,
                calculate_stats=calculate_stats,
                mean=mean,
                std=std,
                normalize=normalize,
                transform=transform,
                target_transform=target_transform,
                transforms=transforms,
                resize=resize,
                normalize_to=normalize_to),
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
        
        :param image_set: The old image set name to split. Accepts both name and ID.
        :type image_set: str | int
        :param new_sets: Each entry of `new_sets` has a name for the set accompanied with a float to
            represent the percentage to split data into.
        :type new_sets: Tuple[str, float]
        :param inplace: Whether to perform the operation inplace on the existing image set. If
            `False`, then the new sets are required to add up to exactly 100% of the compositions.
            If `True`, any remaining percentages less than 100% will be filled back into the old
            image set. Default: `False`.
        :type inplace: bool
        :param seed: The seed to use for the operation, in case consistent dataset manipulation
            in memory is required. Default: `None`
        :type seed: Optional[int]
        '''
        # checks before splitting
        mode = 'name' if isinstance(image_set, str) else 'id'
        check_set = self.image_set_to_idx if mode == 'name' else self.idx_to_image_set
        if image_set not in check_set:
            Warnings.error('image_set_missing', imgset_name=mode, image_set=image_set)
        for new_set in new_sets:
            if new_set[0] in check_set:
                Warnings.error('new_exists', type=mode, imgset_name=new_set[0])

        tot_frac = sum(new_set[1] for new_set in new_sets)
        if not inplace and tot_frac != 1:
            Warnings.error('split_invalid', desc='not equal to')
        if inplace and tot_frac > 1:
            Warnings.error('split_invalid', desc='greater than')

        # assemble new sets
        new_sets: list = list(new_sets)
        if inplace:
            new_sets.append((image_set, 1 - tot_frac))
        if seed is not None:
            random.seed(seed)

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
            if image_set not in row['IMAGE_SET_NAME']:
                continue
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
        if inplace:
            self.clear_image_sets(image_set)

    def get_image_set(
        self,
        image_set: Union[str, int]
    ) -> DataFrame:
        '''
        Retrieve the sub-DataFrame which contains all images in a specific image set.

        :param image_set: The image set. Accepts both string and int.
        :type image_set: str | int
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

        :param sets: If defined, only scan the provided list, otherwise scan all sets.
            Default: `None`.
        :type sets: list[str | int], Optional
        '''
        to_pop = []
        if sets is None:
            sets = self.image_set_to_idx
        for image_set in sets:
            if len(self.get_image_set(image_set)) == 0:
                to_pop.append(image_set)
        for image_set in to_pop:
            if isinstance(image_set, str):
                index = self.image_set_to_idx.pop(image_set, None)
                if index is not None:
                    self.idx_to_image_set.pop(index, None)
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

        :param image_set: The image set to delete. Accepts both name and ID.
        :type image_set: str | int
        '''
        using_id: bool = isinstance(image_set, int)
        if using_id:
            if image_set not in self.idx_to_image_set:
                Warnings.error('image_set_missing', imgset_name='ID', image_set=image_set)
            idx = image_set
            name = self.idx_to_image_set[idx]
        else:
            if image_set not in self.image_set_to_idx:
                Warnings.error('image_set_missing', imgset_name='name', image_set=image_set)
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
        filename: str = '',
        overwrite: bool = False,
        safe: bool = True
    ) -> None:
        '''
        Save the dataset into DynamicData json format.

        :param filename: The filename to save the dataset.
        :type filename: str
        :param overwrite: Whether to overwrite the file if it already exists. Default: `False`.
        :type overwrite: bool
        :param safe: If `True`, do not encode `form` with jsonpickle. Then dataset cannot be
            re-parsed, but is no longer subject to arbitrary code injection upon load.
        :type safe: bool
        '''
        if not safe:
            Warnings.warn('unsafe_save')
        this = {
            'root': self.root,
            'safe': safe,
            'form': None if safe else jsonpickle.encode(self.form, keys=True),
            'dataframe': self.dataframe.to_json(),
            'image_set_to_idx': self.image_set_to_idx,
            'idx_to_image_set': self.idx_to_image_set,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'seg_class_to_idx': self.seg_class_to_idx,
            'idx_to_seg_class': self.idx_to_seg_class,
            'bbox_class_to_idx': self.bbox_class_to_idx,
            'idx_to_bbox_class': self.idx_to_bbox_class,
            'get_md5_hashes': self.get_md5_hashes,
            'bbox_scale_option': self.bbox_scale_option,
            'seg_scale_option': self.seg_scale_option,
            'available_modes': self.available_modes,
            'cleaned': self.cleaned,
        }
        if os.path.exists(filename) and not overwrite:
            Warnings.error('file_exists', filename=filename)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json.dumps(this))

    @classmethod
    def load(cls, filename: str = '') -> Self:
        '''
        Load a DynamicData object from file. Warning: do not load any json files that you did not
        create. This method uses jsonpickle, an insecure loading system with potential for arbitrary
        Python code execution.

        :param filename: The filename to load the data from.
        :type filename: str
        '''
        try:
            print('[DynamicData] Loading dataset...')
            start = time.time()
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not data['safe']:
                Warnings.warn('unsafe_load')
            this: DynamicData = cls(
                data['root'],
                jsonpickle.decode(data['form'], keys=True) if not data['safe'] else None,
                get_md5_hashes=data['get_md5_hashes'],
                bbox_scale_option=data['bbox_scale_option'],
                seg_scale_option=data['seg_scale_option']
            )
            this.dataframe = DataFrame.from_dict(json.loads(data['dataframe']))
            for name in ('image_set', 'class', 'bbox_class', 'seg_class'):
                setattr(
                    this,
                    f'{name}_to_idx',
                    data[f'{name}_to_idx']
                )
                setattr(
                    this,
                    f'idx_to_{name}',
                    {int(i): v for i, v in data[f'idx_to_{name}'].items()}
                )
            this.available_modes = data['available_modes']
            this.cleaned = data['cleaned']
            end = time.time()
            print(f'[DynamicData] Loaded dataset! ({end - start}s)')
        except Exception as e:
            print(f'The following error occurred: \n{e}')
            Warnings.error('invalid_dataset')
        return this

    def sample_image(
        self,
        dpi: float = 1200,
        mode: Optional[str | list[str]] = None,
        idx: Optional[int] = None
    ) -> None:
        '''
        Sample an image from the dataset.

        :param dpi: The image display size, if not in segmentation mode.
        :type dpi: float
        :param mode: Pick from any of the available modes, or supply a list of modes. Default:
            all modes.
        :type mode: Optional[str | list[str]]
        :param idx: Use a specific idx from the dataset. Default: a random image.
        :type idx: Optional[int]
        '''
        if not self.cleaned:
            self.parse()
        if mode is not None:
            mode = union(mode)
            for try_mode in mode:
                if try_mode not in self.available_modes:
                    Warnings.error('mode_unavailable', mode=try_mode)
        else: mode = self.available_modes
        item = self.dataframe.iloc[idx] if idx is not None else self.dataframe.sample().iloc[0]
        transform = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.uint8)
        ])
        image = transform(open_image(item['ABSOLUTE_FILE']).convert('RGB'))
        plt.figure(dpi=dpi)
        if 'classification' in mode:
            print(f'[DynamicData] Image Class ID/Name: {item["CLASS_ID"]}/{item["CLASS_NAME"]}')
        if 'detection' in mode:
            if len(item['BOX']) != 0:
                image = draw_bounding_boxes(
                    image,
                    torch.stack([FloatTensor(box) for box in item['BOX']]),
                    width=3,
                    labels=item['BBOX_CLASS_NAME'],
                    colors='red'
                )
            else: print('[DynamicData] Warning: Image has no bounding boxes.')
        if 'segmentation_mask' in mode:
            _, axarr = plt.subplots(ncols=2)
            axarr[0].imshow(image.permute(1, 2, 0))
            mask = F.to_tensor(open_image(item['ABSOLUTE_FILE_SEG']))
            axarr[1].imshow(mask.permute(1, 2, 0))
        if 'segmentation_poly' in mode:
            _, axarr = plt.subplots(ncols=2)
            axarr[0].imshow(image.permute(1, 2, 0))
            assert len(item['POLYGON']) == len(item['SEG_CLASS_ID']), \
                'SEG_CLASS_ID and POLYGON len mismatch'
            mask = np.asarray(cv2.imread(item['ABSOLUTE_FILE'], cv2.IMREAD_GRAYSCALE))
            mask = np.asarray(mask, dtype=np.int32)
            mask = np.full_like(mask, next_avail_id(self.idx_to_seg_class))
            for class_id, polygon in zip(item['SEG_CLASS_ID'], item['POLYGON']):
                mask = cv2.fillPoly(mask, pts=[np.asarray(polygon, dtype=np.int32)],
                                color=class_id)
            mask = torch.from_numpy(np.asarray(mask))
            axarr[1].imshow(mask)
        else:
            plt.imshow(image.permute(1, 2, 0))
