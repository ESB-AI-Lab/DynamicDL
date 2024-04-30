import torch
from trainer import ClassificationTrainer
from src import DataTypes as DT
from src import *

if __name__ == '__main__':
    batch_size: int = 4
    learning_rate: float = 2.5e-4
    epochs: int = 4

    root = '/Users/atong/Documents/Datasets/OxfordPets'
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
    print(cvdata.idx_to_class)
    cvdata.split_image_set('trainval', ('train', 0.8), ('val', 0.2), inplace = True, seed = 0)
    trainloader = cvdata.get_dataloader('classification',
                                        image_set='train',
                                        batch_size=batch_size,
                                        store_dim=False,
                                        transforms=CVTransforms.CLASSIFICATION)
    valloader = cvdata.get_dataloader('classification',
                                      image_set='val',
                                      batch_size=batch_size,
                                        store_dim=False,
                                      transforms=CVTransforms.CLASSIFICATION)
    testloader = cvdata.get_dataloader('classification',
                                       image_set='test',
                                       batch_size=batch_size,
                                        store_dim=False,
                                       transforms=CVTransforms.CLASSIFICATION)

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
