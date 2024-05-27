from typing import Callable
import torch

from .._warnings import Warnings

def _collate_classification(batch):
    images, labels = zip(*batch)
    try:
        return torch.stack(images), torch.IntTensor(list(labels))
    except RuntimeError as e:
        if 'stack expects each tensor to be equal size, but got' in str(e):
            Warnings.error('invalid_shape', mode='classification')
        raise

def _collate_classification_dim(batch):
    images, labels = zip(*batch)
    labels = {'label': torch.IntTensor([label['label'] for label in labels]),
              'dim': [label['dim'] for label in labels]}
    try:
        return torch.stack(images), labels
    except RuntimeError as e:
        if 'stack expects each tensor to be equal size, but got' in str(e):
            Warnings.error('invalid_shape', mode='classification')
        raise

def _collate_detection(batch):
    return list(zip(*batch))

def _collate_detection_dim(batch):
    images, labels = zip(*batch)
    labels = {'label': torch.IntTensor([label['label'] for label in labels]),
              'dim': [label['dim'] for label in labels]}
    return tuple(images), tuple(labels)

def _collate_segmentation(batch):
    images, labels = zip(*batch)
    try:
        return torch.stack(images), torch.stack(labels)
    except RuntimeError as e:
        if 'stack expects each tensor to be equal size, but got' in str(e):
            Warnings.error('invalid_shape', mode='segmentation')
        raise

def _collate_segmentation_dim(batch):
    images, labels = zip(*batch)
    labels = {'label': torch.stack([label['label'] for label in labels]),
              'dim': [label['dim'] for label in labels]}
    try:
        return torch.stack(images), labels
    except RuntimeError as e:
        if 'stack expects each tensor to be equal size, but got' in str(e):
            Warnings.error('invalid_shape', mode='segmentation')
        raise

def _collate_default(batch):
    return list(zip(*batch))

def _collate(mode: str, store_dim: bool) -> Callable:
    '''
    Getters for all preset collate functions.
    '''
    if mode == 'classification':
        return _collate_classification_dim if store_dim else _collate_classification
    if mode == 'segmentation':
        return _collate_segmentation_dim if store_dim else _collate_segmentation
    if mode == 'detection':
        return _collate_detection_dim if store_dim else _collate_detection
    return _collate_default
