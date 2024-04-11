from trainer import ClassificationTrainer
print("Imported Trainer")
from src import *
print("Imported Dataloader")
from torchvision.models import VGG16_Weights as base
import torchvision.transforms as transforms

if __name__ == '__main__':
    # Number of channels in the image. $3$ for RGB.
    image_channels: int = 3
    # Number of channels in the output mask. $1$ for binary mask.
    mask_channels: int = 1

    # Batch size
    batch_size: int = 4
    # Learning rate
    learning_rate: float = 2.5e-4

    epochs: int = 4

    transform = transforms.Compose([
        base.DEFAULT.transforms()
    ])

    root = '/Users/atong/Documents/Datasets/OxfordPets'
    alias = Alias([
        Generic("{}", DataTypes.IMAGE_NAME),
        Generic("{}_{}", DataTypes.CLASS_NAME, DataTypes.GENERIC)
    ])
    form = {
        Static("annotations"): {
            Generic("{}.txt", DataTypes.IMAGE_SET_NAME): TXTFile(
                GenericList(Generic(
                    "{} {} {} {}", alias, DataTypes.CLASS_ID, DataTypes.GENERIC, DataTypes.GENERIC
                )),
                ignore_type = '#'
            ),
            "trimaps": {
                Generic("{}.png", DataTypes.IMAGE_NAME, ignore='._{}'): SegmentationImage()
            },
            "xmls": {
                Generic("{}.xml", DataTypes.IMAGE_NAME): XMLFile({
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
        Static("images"): {
            Generic("{}.jpg", alias): Image()
        }
    }
    cvdata = CVData(root, form, transform=base.DEFAULT.transforms(), batch_size = 4)
    cvdata.cleanup()
    cvdata.split_image_set('trainval', ('train', 0.8), ('val', 0.2), inplace = True, seed = 0)
    trainloader = cvdata.get_dataloader('classification', 'train')
    valloader = cvdata.get_dataloader('classification', 'val')
    testloader = cvdata.get_dataloader('classification', 'test')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

