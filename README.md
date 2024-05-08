# CVData

## Introduction

In recent days computer vision has become a hot topic in the field of AI and computer science. Advances in hardware and computational power has enabled deep learning to become pervasive in everyday life, and computer vision architectures serve many purposes in today's world. Yet a high barrier of entry remains in dataloading. While deep learning architectures are a "plug and chug", requiring users to possess near-zero knowledge about its inner works, dataloading still remains a barrier to most seeking to test their custom datasets. CVData is a library to counteract that, containing code to dynamically process datasets.

## Tutorials found [here!](./tutorial.md) 
## Docs found [here!](https://atong28.github.io/CVData/)

## Installation

Building from dev branch, navigate to the clone and run

```
$ pip install .
```

From PyPI:

```
$ pip install cvdata
```

For server versions use `cvdata-headless` instead of `cvdata`.

```python
import torch
from cvdata import DataTypes as DT
from cvdata import *

if __name__ == '__main__':
    batch_size: int = 4
    learning_rate: float = 2.5e-4
    epochs: int = 4

    root = 'MY_ROOT_DIR'
    alias = Alias([
        Generic(DT.IMAGE_NAME),
        Generic("{}_{}", DT.CLASS_NAME, DT.GENERIC)
    ])
    form = {
        "annotations": {
            File("{}", DT.IMAGE_SET_NAME, extensions="txt"): TXTFile(
                GenericList(Generic(
                    "{} {} {} {}", alias, DT.CLASS_ID, DT.GENERIC, DT.GENERIC
                )),
                ignore_type = '#'
            ),
            "trimaps": {
                ImageFile("{}", DT.IMAGE_NAME, ignore='._{}'): SegmentationImage()
            },
            "xmls": {
                File("{}", DT.IMAGE_NAME, extensions='xml'): XMLFile({
                    "annotation": {
                        "filename": Generic("{}.jpg", DT.IMAGE_NAME),
                        "object": AmbiguousList({
                            "name": DT.BBOX_CLASS_NAME,
                            "bndbox": {
                                "xmin": DT.XMIN,
                                "ymin": DT.YMIN,
                                "xmax": DT.XMAX,
                                "ymax": DT.YMAX
                            }
                        })
                    }
                })
            }
        },
        "images": {ImageFile(alias): Image()}
    }
    cvdata = CVData(root, form)
    cvdata.parse()
    cvdata.split_image_set('trainval', ('train', 0.8), ('val', 0.2), inplace = True, seed = 0)
    trainloader = cvdata.get_dataloader(
        'classification',
        image_set='test',
        batch_size=batch_size,
        store_dim=False,
        preset_transform=True
    )
    valloader = cvdata.get_dataloader(
        'classification',
        image_set='test',
        batch_size=batch_size,
        store_dim=False,
        preset_transform=True
    )
    testloader = cvdata.get_dataloader(
        'classification',
        image_set='test',
        batch_size=batch_size,
        store_dim=False,
        preset_transform=True
    )
```