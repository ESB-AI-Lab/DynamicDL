'''
Adapted from torchvision:
https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py
'''

from typing import Optional, Tuple, Callable
import torch
from torch import nn, Tensor
from torchvision.transforms import functional as F, InterpolationMode


class ObjectDetection(nn.Module):
    def __init__(
        self,
        normalize: bool = True,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def forward(self, img: Tensor) -> Tensor:
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        if self.normalize:
            img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image "
            "``torch.Tensor`` objects. The images are rescaled to ``[0.0, 1.0]``."
        )

class ImageClassification(nn.Module):
    def __init__(
        self,
        *,
        crop_size: int,
        resize_size: int = 256,
        normalize: bool = True,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.crop_size = [crop_size]
        self.resize_size = [resize_size]
        self.normalize = normalize
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img: Tensor) -> Tensor:
        img = F.resize(img, self.resize_size, interpolation=self.interpolation, antialias=self.antialias)
        img = F.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        if self.normalize:
            img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image "
            "``torch.Tensor`` objects. The images are resized to "
            f"``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
            f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are "
            f"first rescaled to ``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and "
            f"``std={self.std}``."
        )

class SemanticSegmentation(nn.Module):
    def __init__(
        self,
        *,
        resize_size: Optional[int],
        normalize: Optional[bool],
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.resize_size = [resize_size] if resize_size is not None else None
        self.normalize = normalize
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img: Tensor) -> Tensor:
        if isinstance(self.resize_size, list):
            img = F.resize(
                img,
                self.resize_size,
                interpolation=self.interpolation,
                antialias=self.antialias
            )
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        if self.normalize:
            img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image "
            "``torch.Tensor`` objects. The images are resized to "
            f"``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``. "
            f"Finally the values are first rescaled to ``[0.0, 1.0]`` and then normalized using "
            f"``mean={self.mean}`` and ``std={self.std}``."
        )

class Transforms:
    '''
    Standard transforms presets for computer vision. Adapted from torchvision 0.17.2
    '''
    CLASSIFICATION = (ImageClassification(crop_size=224), None)
    DETECTION = (ObjectDetection(), None)
    SEGMENTATION = (SemanticSegmentation(resize_size=520, normalize=True),
                    SemanticSegmentation(resize_size=520, normalize=False))
    SEGMENTATION_NORESIZE = (SemanticSegmentation(resize_size=None, normalize=True),
                             SemanticSegmentation(resize_size=None, normalize=False))

    @staticmethod
    def get(
        mode: str,
        resize: bool = False,
        normalize: bool = True,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ) -> Tuple[Optional[Callable], ...]:
        '''
        Retrieve the standard transforms for classification, detection, and segmentation.
        
        - `mode` (`str`): choose between classification, detection, and segmentation.
        - `resize` (`bool`): True if planning to resize. Default: False.
        - `normalize` (`bool`): True for normalized transforms according to mean and std, False
            otherwise. Default: True.
        - `mean` (`Tuple[float, ...]`): the mean to use for normalization. Has no effect when
            normalize is false. Default is ImageNet dataset statistics.
        - `std` (`Tuple[float, ...]`): the std to use for normalization. Has no effect when
            normalize is false. Default is ImageNet dataset statistics.
        '''
        if mode == 'classification':
            return (
                ImageClassification(crop_size=224, normalize=normalize, mean=mean, std=std),
                None
            )
        if mode == 'detection':
            return (ObjectDetection(normalize=normalize, mean=mean, std=std), None)
        if mode == 'segmentation':
            resize = None if resize else 520
            return (
                SemanticSegmentation(resize_size=resize, normalize=normalize, mean=mean, std=std),
                SemanticSegmentation(resize_size=resize, normalize=False, mean=mean, std=std)
            )
        return (None, None)
