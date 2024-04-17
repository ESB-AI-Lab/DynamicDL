import torch
from trainer import ClassificationTrainer
from src import *

if __name__ == '__main__':
    batch_size: int = 4
    learning_rate: float = 2.5e-4
    epochs: int = 4

    root = '/Users/atong/Documents/Datasets/OxfordPets'
    alias = Alias([
        Generic(DataTypes.IMAGE_NAME),
        Generic("{}_{}", DataTypes.CLASS_NAME, DataTypes.GENERIC)
    ])
    form = {
        "annotations": {
            File("{}.txt", DataTypes.IMAGE_SET_NAME): TXTFile(
                GenericList(Generic(
                    "{} {} {} {}", alias, DataTypes.CLASS_ID, DataTypes.GENERIC, DataTypes.GENERIC
                )),
                ignore_type = '#'
            ),
            "trimaps": {
                File("{}.png", DataTypes.IMAGE_NAME, ignore='._{}'): SegmentationImage()
            },
            "xmls": {
                File("{}.xml", DataTypes.IMAGE_NAME): XMLFile({
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
        "images": {
            File("{}.jpg", alias): Image()
        }
    }
    cvdata = CVData(root, form)
    cvdata.parse()
    cvdata.split_image_set('trainval', ('train', 0.8), ('val', 0.2), inplace = True, seed = 0)
    trainloader = cvdata.get_dataloader('classification',
                                        'train',
                                        batch_size=batch_size,
                                        transforms=CVTransforms.CLASSIFICATION_TRANSFORMS)
    valloader = cvdata.get_dataloader('classification',
                                      'val',
                                      batch_size=batch_size,
                                      transforms=CVTransforms.CLASSIFICATION_TRANSFORMS)
    testloader = cvdata.get_dataloader('classification',
                                       'test',
                                       batch_size=batch_size,
                                       transforms=CVTransforms.CLASSIFICATION_TRANSFORMS)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    config = {
        'device': device,
        'model_args': {
            'name': 'vgg16',
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
            'classes': cvdata.class_to_idx
        },
        'checkpointing': True,
        'num_epochs': 25
    }
    trainer = ClassificationTrainer.from_config(config)
    trainer.do_training('run_1_cls')
