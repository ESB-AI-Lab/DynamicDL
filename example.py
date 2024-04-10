####
# COCODetection
####

from src import *
root = 'MY_ROOT_DIR'
form = {
    Generic('{}.json', DataTypes.IMAGE_SET_NAME): JSONFile({
        'images': GenericList([{
            'id': DataTypes.IMAGE_ID,
            'file_name': Generic('{}.jpg', DataTypes.IMAGE_NAME)
        }]),
        'categories': Pairing(GenericList([{
            'id': DataTypes.BBOX_CLASS_ID,
            'name': DataTypes.BBOX_CLASS_NAME
        }]), DataTypes.BBOX_CLASS_ID, DataTypes.BBOX_CLASS_NAME),
        'annotations': GenericList([{
            'image_id': DataTypes.IMAGE_ID,
            'category_id': DataTypes.BBOX_CLASS_ID,
            'bbox': GenericList([
                DataTypes.X1, DataTypes.Y1, DataTypes.X2, DataTypes.Y2
            ])
        }])
    }),
    'images': {
        Generic('{}.jpg', DataTypes.IMAGE_NAME): Image()
    }
}
cvdata = CVData(root, form)
cvdata.cleanup()

####
# CocoSegmentation
####

from src import *
root = 'MY_ROOT_DIR'
form = {
    Generic('{}.json', DataTypes.IMAGE_SET_NAME): JSONFile({
        'images': GenericList([{
            'id': DataTypes.IMAGE_ID,
            'file_name': Generic('{}.jpg', DataTypes.IMAGE_NAME)
        }]),
        'categories': Pairing(GenericList([{
            'id': DataTypes.SEG_CLASS_ID,
            'name': DataTypes.SEG_CLASS_NAME
        }]), DataTypes.SEG_CLASS_ID, DataTypes.SEG_CLASS_NAME),
        'annotations': GenericList([{
            'image_id': DataTypes.IMAGE_ID,
            'category_id': DataTypes.SEG_CLASS_ID,
            'segmentation': GenericList(SegmentationObject([
                DataTypes.X, DataTypes.Y
            ]))
        }])
    }),
    'images': {
        Generic('{}.jpg', DataTypes.IMAGE_NAME): Image()
    }
}
cvdata = CVData(root, form)
cvdata.cleanup()

####
# OxfordPets
####

from src import *
root = 'MY_ROOT_DIR'
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
            Generic("{}.png", DataTypes.IMAGE_NAME): SegmentationImage()
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
cvdata = CVData(root, form)
cvdata.cleanup()
