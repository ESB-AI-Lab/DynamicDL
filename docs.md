# CVData Documentation

## Quick Links

### CVData Main Module
 - [`CVData`](#cvdata) (main data module)
 - `CVDataset` (dataset module)

### Basic Data Classes
 - `DataType` (data types, psuedo-private)
 - `DataTypes` (presets for data types)
 - `DataItem` (data items containing values of specific data types)
 - `DataEntry` (combinations of associated data items)
 - `Alias` (combination of data types in same generic pattern)
 - `Static` (represents one item, possibly with data)
 - `Generic` (represents a generic item, with data)
 - `Folder` (folder/directory generic)
 - `File` (file generic)
 - `ImageFile` (image file generic)
 - `Image` (image file token)
 - `SegmentationImage` (segmentation image file token)
 - `GenericList` (list items)
 - `SegmentationObject` (collection of x, y coords for segmentation polygon)
 - `AmbiguousList` (use in xml file when data is ambiguously a list or a single item)
 - `Pairing` (associate data types together)

### File Processing Classes
 - `JSONFile` (json parser)
 - `TXTFile` (txt parser)
 - `XMLFile` (xml parser)
 - `YAMLFile` (yaml parser, untested)

## Documentation

### CVData

```python
class CVData
```

Main dataset class. Accepts root directory path and dictionary form of the structure.
    
Constructor Args:
- `root` (`str`): the root directory to access the dataset.
- `form` (`dict`): the form of the dataset. See documentation for further details on valid
forms.
- `bbox_scale_option` (`str`): choose from either 'auto', 'zeroone', or 'full' scale options to define, or leave empty for automatic. Default: 'auto'
- `seg_scale_option` (`str`): choose from either 'auto', 'zeroone', or 'full' scale options to define, or leave empty for automatic. Default: 'auto'
- `get_md5_hashes` (`bool`): when set to True, create a new column which finds md5 hashes for each image available, and makes sure there are no duplicates. Default: False
- `purge_duplicates` (`Optional[bool]`): when set to True, remove all duplicate image entries. Duplicate images are defined by having the same md5 hash, so this has no effect when get_md5_hashes is False. When set to False, do not purge duplicates. Default: None

```python
def parse(self, override: bool = False) -> None
```
Must be called to instantiate the data in the dataset instance. Performs the recursive populate_data algorithm and creates the dataframe, and then cleans up the data.

Args:
- `override` (`bool`): whether to overwrite existing data if it has already been parsed and cleaned. Default: False

```python
def get_transforms(self, mode: str, remove_invalid: bool = True, resize: Optional[tuple[int, int]] = None) -> tuple[Callable, Callable]
```

```python
def get_dataset(self, mode: str, remove_invalid: bool = True, store_dim: bool = False, preset_transform: bool = True, calculate_stats: bool = True, image_set: Optional[Union[int, str]] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[tuple[Callable]] = None, resize: Optional[tuple[int, int]] = None, normalize: Optional[str] = None) -> CVDataset:
```

Retrieve the dataset (`CVDataset`) of a specific mode and image set.
        
Args:
- `mode` (`str`): the mode of training to select. See available modes with
`available_modes`.
- `remove_invalid` (`bool`): if set to True, deletes any NaN/corrupt items in the image set
pertaining to the relevant mode. In the False case, either NaN values are substituted
with empty values or an error is thrown, depending on the mode selected.
- `store_dim` (`bool`): if set to True, the labels in the dataset will return a dict with
two keys. 'label' contains the standard PyTorch labels and 'dim' contains the image's
former dimensions.
- `preset_transform` (`bool`): whether to use default preset transforms. Consists of
normalization with either calculated mean of the dataset about to be used or standard
ImageNet statistics depending on `calculate_stats`. Default: True
- `calculate_stats` (`bool`): whether to calculate mean and std for this dataset to be used
in normalization transforms. If False, uses ImageNet default weights. Default: True
- `image_set` (`Optional[str]`): the image set to pull from. Default: all images.
- `transform` (`Optional[Callable]`): the transform operation to apply to the images.
- `target_transform` (`Optional[Callable]`): the transform operation on the labels.
- `transforms` (`Optional[tuple[Optional[Callable], ...]]`): tuple in the format
`(transform, target_transform)`. Default PyTorch transforms are available in the
CVTransforms class.
- `resize` (`Optional[tuple[int, int]]`): if provided, resize all images to exact
configuration.
- `normalize` (`Optional[str]`): if provided, normalize bounding box/segmentation
coordinates to a specific configuration. Options: 'zeroone', 'full'

```python
def get_dataloader(self, mode: str, batch_size: int = 4, shuffle: bool = True, num_workers: int = 1, remove_invalid: bool = True, store_dim: bool = False, preset_transform: bool = True, image_set: Optional[Union[int, str]] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[tuple[Callable]] = None, resize: Optional[tuple[int, int]] = None, normalize: Optional[str] = None) -> torch.utils.data.DataLoader
```

Retrieve the PyTorch dataloader (`torch.utils.data.DataLoader`) for this dataset.

Args:
- `mode` (`str`): the mode of training to select. See available modes with
`available_modes`.
- `batch_size` (`int`): the batch size of the image. Default: 4.
- `shuffle` (`bool`): whether to shuffle the data before loading. Default: True.
- `num_workers` (`int`): number of workers for the dataloader. Default: 1.
- `remove_invalid` (`bool`): if set to True, deletes any NaN/corrupt items in the image set pertaining to the relevant mode. In the False case, either NaN values are substituted with empty values or an error is thrown, depending on the mode selected.
- `store_dim` (`bool`): if set to True, the labels in the dataset will return a dict withtwo keys. 'label' contains the standard PyTorch labels and 'dim' contains the image's former dimensions.
- `preset_transform` (`bool`): whether to use default preset transforms. Consists of normalization with either calculated mean of the dataset about to be used or standard ImageNet statistics depending on `calculate_stats`. Default: True
- `calculate_stats` (`bool`): whether to calculate mean and std for this dataset to be used in normalization transforms. If False, uses ImageNet default weights. Default: True
- `image_set` (`Optional[str]`): the image set to pull from. Default: all images.
- `transform` (`Optional[Callable]`): the transform operation to apply to the images.
- `target_transform` (`Optional[Callable]`): the transform operation on the labels.
- `transforms` (`Optional[tuple[Optional[Callable], ...]]`): tuple in the format
`(transform, target_transform)`. Default PyTorch transforms are available in the CVTransforms class.
- `resize` (`Optional[tuple[int, int]]`): if provided, resize all images to exact configuration.
- `normalize` (`Optional[str]`): if provided, normalize bounding box/segmentation coordinates to a specific configuration. Options: 'zeroone', 'full'

```python
def split_image_set(self, image_set: Union[str, int], *new_sets: tuple[str, float], inplace: bool = False, seed: Optional[int] = None) -> None:
```

Split the existing image set into new image sets. If inplace is True, the existing image
set will receive the percentage that is missing from the rest of the sets, or deleted if
the other sets add up to 1.

Args:
- `image_set` (`str | int`): the old image set name to split. Accepts both name and ID.
- `new_sets` (`tuple[str, float]`): each entry of new_sets has a name for the set accompanied with a float to represent the percentage to split data into.
- `inplace` (`bool`): whether to perform the operation inplace on the existing image set. If False, then the new sets are required to add up to exactly 100% of the compositions. If True, any remaining percentages less than 100% will be filled back into the old image set. Default: False.
- `seed` (`Optional[int]`): the seed to use for the operation, in case consistent dataset manipulation in memory is required. Default: None

```python
def get_image_set(self, image_set: Union[str, int]) -> DataFrame
```
Retrieve the sub-DataFrame which contains all images in a specific image set.

Args:
- `image_set` (`str | int`): the image set. Accepts both name and ID.