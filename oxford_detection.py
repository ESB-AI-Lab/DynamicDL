import torch
from trainer import DetectionTrainer
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
    cvdata.delete_image_set('test')
    #print(cvdata.image_set_to_idx)
    cvdata.split_image_set('trainval', ('train', 0.64), ('val', 0.16), ('test', 0.2), inplace = True, seed = 0)
    trainloader = cvdata.get_dataloader('detection', image_set='train', batch_size=batch_size, transforms=CVTransforms.DETECTION_TRANSFORMS)
    valloader = cvdata.get_dataloader('detection', image_set='val', batch_size=batch_size, transforms=CVTransforms.DETECTION_TRANSFORMS)
    testloader = cvdata.get_dataloader('detection', image_set='test', batch_size=batch_size, transforms=CVTransforms.DETECTION_TRANSFORMS)
    #print(cvdata.bbox_class_to_idx)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device {device}')
    config = {
        'device': device,
        'model_args': {
            'name': 'fasterrcnn_resnet50_fpn_v2',
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
            'classes': cvdata.bbox_class_to_idx
        },
        'checkpointing': True,
        'num_epochs': 25
    }
    trainer = DetectionTrainer.from_config(config)
    trainer.do_training('run_1_bbox')
