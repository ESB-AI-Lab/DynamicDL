from typing import Optional, Callable

import cv2
import numpy as np
from pandas import DataFrame
from pandas.core.series import Series
from torch import FloatTensor, Tensor, LongTensor
import torch
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms.functional as F
from PIL.Image import open as open_image

from ._warnings import Warnings

class DynamicDS(VisionDataset):
    '''
    Dataset implementation for the DynamicData environment.

    :param df: The dataframe from DynamicData.
    :type df: DataFrame
    :param root: The root of the dataset folder.
    :type root: str
    :param mode: The mode of the data to retrieve, i.e. classification, segmentation, etc.
    :type mode: str
    :param id_mapping: The id mapping from the dataframe to retrieve class names. This is used
        primarily as a safety feature in order to make sure that used IDs are provided in order
        starting from 0 without holes so that training works properly.
    :type id_mapping: dict[int, int]
    :param image_type: The type of the image to export, to convert PIL images to.
            Default: 'RGB'. Also accepts 'L' and 'CMYK'.
    :type image_type: str
    :param normalization: The type of normalization that the dataset currently is formatted in,
        for box and polygon items. Accepts 'full' or 'zeroone'.
    :type normalization: str
    :param normalize_to: The type of normalization that the dataset is to be resized to, for
        box and polygon items. Accepts 'full' or 'zeroone'.
    :type normalize_to: str
    :param transform: The transform operation to apply to the images.
    :type transform: Optional[Callable]
    :param target_transform: The transform operation on the labels.
    :type target_transform: Optional[Callable]
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
        if self.mode == 'segmentation':
            self.default = len(self.id_mapping)
        super().__init__(
            root,
            transforms=None,
            transform=transform,
            target_transform=target_transform
        )

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
        if self.resize:
            factor_resize = self.resize
        else:
            factor_resize = (1, 1)
        if self.normalization == 'full':
            factor_norm = item['IMAGE_DIM']
        else:
            factor_norm = (1, 1)
        apply_resize = lambda p: (p[0] * factor_resize[0] / factor_norm[0],
                                  p[1] * factor_resize[1] / factor_norm[1],
                                  p[2] * factor_resize[0] / factor_norm[0],
                                  p[3] * factor_resize[1] / factor_norm[1])
        bbox_tensors = [FloatTensor(apply_resize(box)) for box in boxes]
        if not bbox_tensors:
            Warnings.error('empty_bbox', file=item['ABSOLUTE_FILE'])
        if self.store_dim:
            return {
                'label': {'boxes': torch.stack(bbox_tensors), 'labels': LongTensor(class_ids)},
                'dim': item['IMAGE_DIM']
            }
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
            mask = np.full(self.resize, self.default, dtype=np.int32)
            factor_resize = self.resize
        else:
            mask = np.full(item['IMAGE_DIM'], self.default, dtype=np.int32)
            factor_resize = (1, 1)
        if self.normalization == 'full':
            factor_norm = item['IMAGE_DIM']
        else: factor_norm = (1, 1)
        apply_resize = lambda p: (p[0] * factor_resize[0] / factor_norm[0],
                                  p[1] * factor_resize[1] / factor_norm[1])
        for class_id, polygon in zip(item['SEG_CLASS_ID'], item['POLYGON']):
            if self.resize is not None:
                polygon = list(map(apply_resize, polygon))
            mask = cv2.fillPoly(mask, pts=[np.asarray(polygon, dtype=np.int32)],
                            color=self.id_mapping[class_id])
        mask = torch.from_numpy(np.asarray(mask)).unsqueeze(-1).permute(2, 0, 1)
        return mask

    def __getitem__(self, idx):
        item: dict = self.data[idx]
        image: Tensor = F.to_tensor(open_image(item.get('ABSOLUTE_FILE')).convert(self.image_type))
        if self.resize:
            image = F.resize(image, [self.resize[1], self.resize[0]])
        label: dict[str, Tensor]
        if self.mode in {'inference', 'diffusion'}:
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
