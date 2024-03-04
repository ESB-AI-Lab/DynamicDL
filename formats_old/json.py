'''
Coco detection format based on AvoVision dataset

from tokens import Token

format = {
    "label": Token.DATASET_ROOT,
    "type": Token.DETECTION_MODE,
    "root": {
        "images": {
            "label": Token.IMAGES,
            "type": Token.DIRECTORY,
            "item": {
                "label": Token.IMAGE,
                "type": Token.IMAGE,
                "filename": Token.IMAGE_FILENAME
            }
        }
        "result.json": {
            "label": Token.ANNOTATION_FILE,
            "type": Token.JSON_FILE,
            "format": "coco_format"
        }
    }
    "formats": {
        "coco_format": {
            "label": Token.ANNOTATION_FILE,
            "type": Token.JSON_FILE,
            "structure": {
                "images": {
                    "label": Token.IMAGES,
                    "type": Token.LIST,
                    "item": {
                        "width": Token.NONE,
                        "height": Token.NONE,
                        "id": Token.IMAGE_IDX,
                        "file_name": Token.IMAGE_FILENAME
                    }
                },
                "categories": {
                    "label": Token.CLASSES,
                    "type": Token.LIST,
                    "item": {
                        "id": Token.CLASS_IDX,
                        "name": Token.CLASS
                    }
                },
                "annotations": {
                    "label": Token.ANNOTATIONS,
                    "type": Token.LIST
                    "item": {
                        "id": Token.NONE,
                        "image_id": Token.IMAGE_IDX,
                        "category_id": Token.CLASS_IDX,
                        "segmentation": Token.NONE,
                        "bbox": [
                            Token.XMIN,
                            Token.YMIN,
                            Token.WIDTH,
                            Token.HEIGHT
                        ],
                        "ignore": Token.NONE,
                        "iscrowd": Token.NONE,
                        "area": Token.NONE
                    }
                }
            }
        }
    }
}




Filestructure specifications:

Valid labels for any folder: 
- ANNOTATIONS: must be a directory with ANNOTATION_FILE items at top level
- IMAGES: must be a directory with IMAGE items at top level
- CLASSES: must be a directory which contains more directories, each their own
           class of some format (ANNOTATIONS, IMAGES, IMAGE_SETS)
- IMAGE_SETS: must be a directory which contains more directories, each their
              own image set of some format (ANNOTATIONS, IMAGES, CLASSES).
              IMAGE_SETS directory type cannot be contained under another
              directory of IMAGE_SETS type (must be unique).
- FILLER: must be a directory which contains more directories, with no specific
          meaning (i.e. you can collapse the directories into the root directory
          but avoided doing so for some form of clarity or organization)

For any folder, it must contain one of the following:
- item: a generic item type format to read from. Folder can only be of the types
        ANNOTATIONS, IMAGES
- items: a specific, finite list of items and their associated formats. Folder
         can only be of the types ANNOTATIONS, IMAGES
- folder: a generic folder type format to read from. Folder can only be of the
          types CLASSES, IMAGE_SETS, FILLER
- folders: a specific, finite list of items and their associated formats. Folder
           can only be of the types CLASSES, IMAGE_SETS, FILLER

Valid format for any generic item format:
- ANNOTATION_FILE: must be an object which specifies filetype (JSON_FILE, etc.),
                   file contents must follow a specified format defined in 
                   formats, and is either contained under an IMAGE_SETS folder
                   or specifies its own image_set
                   
                   must contain a description of which tokens are included in
                   each annotation file (BBOX, SEGMENTATION, CLASS, etc.)
                   
                   the aggregation of all annotation files must fulfill reqs.
                   i.e. must contain valid classes, either through a class file
                   or specified upon initialization
                   
                   verify that for each annotation there is a valid method to
                   determining the corresponding image, either through an
                   IMAGE_IDX system, IMAGE_CLASSIFIER type which is found
                   through the string, or a proper IMAGE_FILENAME is specified.
- IMAGE: must be an object which specifies an individual image filepath, and is
         either contained under an IMAGE_SETS folder or specifies its own
         image_set

Annotation requirements for each item:

Image Classification:
- Each item must have a specified image, depending on some sort of classifier,
  and an associated annotation of CLASS label type.

Object Detection (Bounding Box):
- Each item must have a specified image, depending on some sort of classifier,
  and an associated annotation(s):
  Bounding boxes must include (XMIN, YMIN), and one of (XMAX, YMAX) or 
  (WIDTH, HEIGHT). Rotated bounding boxes will not be supported. Each bounding
  box must be accompanied by a CLASS label type. Bounding boxes must either
  each specify their own image or be part of an item which specifies image.
  
Segmentation (Maps):
- Ignore for now. PyTorch segmentation models seem to be incomplete.

Warnings:
- If image dimensions are provided, throw a warning in Classification mode if
  the images are not of the same dimension.
- If the list of classes contains extra classes than which that is present in
  the dataset, or missing classes which are present in the dataset, throw a
  warning.
- If there is an item detected that is unrecognizable, throw a warning.
'''

class JSON:
    pass