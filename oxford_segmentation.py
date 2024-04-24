import torch
from trainer import SegmentationTrainer
from src import *

if __name__ == '__main__':
    image_channels: int = 3
    mask_channels: int = 1
    batch_size: int = 1
    learning_rate: float = 2.5e-4
    epochs: int = 4

    root = '/Users/atong/Documents/Datasets/OxfordPets'
    alias = Alias([
        Generic(DataTypes.IMAGE_NAME),
        Generic("{}_{}", DataTypes.CLASS_NAME, DataTypes.GENERIC)
    ])
    form = {
        "annotations": {
            File("{}", DataTypes.IMAGE_SET_NAME, extensions="txt"): TXTFile(
                GenericList(Generic(
                    "{} {} {} {}", alias, DataTypes.CLASS_ID, DataTypes.GENERIC, DataTypes.GENERIC
                )),
                ignore_type = '#'
            ),
            "trimaps": {
                ImageFile("{}", DataTypes.IMAGE_NAME, ignore='._{}'): SegmentationImage()
            },
            "xmls": {
                File("{}", DataTypes.IMAGE_NAME, extensions='xml'): XMLFile({
                    "annotation": {
                        "filename": Generic("{}.jpg", DataTypes.IMAGE_NAME),
                        "object": AmbiguousList({
                            "name": DataTypes.BBOX_CLASS_NAME,
                            "bndbox": {
                                "xmin": DataTypes.XMIN,
                                "ymin": DataTypes.YMIN,
                                "xmax": DataTypes.XMAX,
                                "ymax": DataTypes.YMAX
                            }
                        })
                    }
                })
            }
        },
        "images": {ImageFile(alias): Image()}
    }
    cvdata = CVData(root, form)
    # since the oxford pets dataset does not specify seg classes, we will have to do so manually
    cvdata.seg_class_to_idx = {'body': 0, 'outline': 1, 'background': 2}
    cvdata.idx_to_seg_class = {0: 'body', 1: 'outline', 2: 'background'}
    cvdata.parse()
    cvdata.split_image_set('trainval', ('train', 0.8), ('val', 0.2), inplace = True, seed = 0)
    trainloader = cvdata.get_dataloader(
        'segmentation',
        resize=(512, 512),
        image_set='train',
        batch_size=batch_size,
        transforms=CVTransforms.DETECTION_NORESIZE
    )
    valloader = cvdata.get_dataloader(
        'segmentation',
        resize=(512, 512),
        image_set='val',
        batch_size=batch_size,
        transforms=CVTransforms.DETECTION_NORESIZE
    )
    testloader = cvdata.get_dataloader(
        'segmentation',
        resize=(512, 512),
        image_set='test',
        batch_size=batch_size,
        transforms=CVTransforms.DETECTION_NORESIZE
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    config = {
        'device': device,
        'model_args': {
            'name': 'deeplabv3_resnet50',
            'weight_type': None
        },
        'optimizer_args': {
            'name': 'SGD',
            'lr': learning_rate
        },
        'dataloaders': {
            'train': trainloader,
            'test': testloader,
            'val': valloader,
            'classes': cvdata.seg_class_to_idx
        },
        'checkpointing': True,
        'num_epochs': 25
    }
    trainer = SegmentationTrainer.from_config(config)
    trainer.do_training('run_1_seg')
