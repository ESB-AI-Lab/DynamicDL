# CVData Tutorial

CVData uses a hierarchy-based parsing algorithm to take in data. Given some dataset, you need to feed in the *structure* of the dataset.

Suppose you have a standard COCODetection dataset with the following structure:

```
coco_dataset/
|   
├── annotations/
|   ├── instances_train.json
|   └── instances_val.json
|
└── images/
    ├── train/
    |   ├── image_train1.jpg
    |   ├── image_train2.jpg
    |   └── ...
    |
    └── val/
        ├── image_val1.jpg
        ├── image_val2.jpg
        └── ...
```

You will need to be able to tell CVData what each item does, generically. An example for a possible syntax in python uses `dict` objects:

```python
import cvdata as cvd
root = '~/coco_dataset/'
form = {
    'annotations': {
        'instances_train.json': cvd.JSONFile( ... ),
        'instances_val.json': cvd.JSONFile( ... )
    },
    'images': {
        'train': {
            'image_train1.jpg': cvd.Image(),
            'image_train2.jpg': cvd.Image(),
            ...
        },
        'val': {
            'image_val1.jpg': cvd.Image(),
            'image_val2.jpg': cvd.Image(),
            ...
        }
    }
}
```

But the dataset has thousands of images! We can't possibly write all of them manually. Also, we need a method to tell `cvdata` how to extract data from both the names of the files, folders, and also the contents of the files. 

The `DataTypes` class introduces a solution: in the class we have many preset tokens which cover the functionalities of the image classification, object detection, and semantic segmentation operations. More on this in the [documentation](./docs.md). Then, we can wrap these `DataType` objects into `Generic` objects (or `File` and `Folder` objects, which extend the `Generic` class)! Let's update the previous example:

```python
from cvdata import DataTypes as DT
import cvdata as cvd
root = '~/coco_dataset/'
form = {
    'annotations': {
        cvd.File('instances_{}', DT.IMAGE_SET_NAME): cvd.JSONFile( ... )
    },
    'images': {
        cvd.Folder(DT.IMAGE_SET_NAME): {
            cvd.ImageFile(DT.IMAGE_NAME): cvd.Image()
        }
    }
}
```

Much better. But now we need to find out what is inside the JSON file! The syntax remains nearly the same. For a COCO dataset, here's what it looks like inside:

###### annotations.json
```json
{
  "info": {
    "description": "COCO Dataset",
    "url": "http://cocodataset.org/",
    "version": "1.0",
    "year": 2024,
    "contributor": "John Doe",
    "date_created": "2024-05-01"
  },
  "licenses": [
    {
      "id": 1,
      "name": "Attribution-NonCommercial",
      "url": "http://creativecommons.org/licenses/by-nc/2.0/"
    }
  ],
  "images": [
    {
      "id": 1,
      "width": 640,
      "height": 480,
      "file_name": "image_train1.jpg",
      "license": 1,
      "flickr_url": "",
      "coco_url": "",
      "date_captured": "2024-05-01"
    },
    {
      "id": 2,
      "width": 800,
      "height": 600,
      "file_name": "image_train2.jpg",
      "license": 1,
      "flickr_url": "",
      "coco_url": "",
      "date_captured": "2024-05-01"
    }
    ...
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [],
      "area": 0,
      "bbox": [0, 0, 0, 0],
      "iscrowd": 0
    },
    {
      "id": 2,
      "image_id": 2,
      "category_id": 2,
      "segmentation": [],
      "area": 0,
      "bbox": [0, 0, 0, 0],
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "category1",
      "supercategory": "object"
    },
    {
      "id": 2,
      "name": "category2",
      "supercategory": "object"
    }
  ]
}
```

That's a lot of data! Fortunately, we only have to tell `CVData` what we need. Here's our form once again, expanded with the `JSONFile` constructor:

```python
from cvdata import DataTypes as DT
import cvdata as cvd
root = '~/coco_dataset/'
form = {
    'annotations': {
        cvd.File('instances_{}', DT.IMAGE_SET_NAME): cvd.JSONFile({
            'images': [{
                'id': DT.IMAGE_ID,
                'file_name': DT.ABSOLUTE_FILE
            }],
            'categories': Pairing([{
                'id': DT.BBOX_CLASS_ID,
                'name': DT.BBOX_CLASS_NAME
            }]),
            'annotations': [{
                'image_id': DT.IMAGE_ID,
                'category_id': DT.BBOX_CLASS_ID,
                'bbox': [DT.XMID, DT.YMID, DT.WIDTH, DT.HEIGHT]
            }]
        })
    },
    'images': {
        cvd.Folder(DT.IMAGE_SET_NAME): {
            cvd.ImageFile(DT.IMAGE_NAME): cvd.Image()
        }
    }
}

cvdata = CVData(root, form)
cvdata.parse()
```

Voila! We have a fully functional COCO Dataset parser. Notice how straightforward it was - the JSONFile parsing almost exactly matches the syntax of the json file itself. The only exception is the `Pairing` class, which is a wrapper class to tell `cvdata` that two datatypes are generally associated together but are not tied to any specific images. This is to avoid confusion and some problems that may arise during parsing.

How does it work? Well, `cvdata` uses a hierarchy-based algorithm to merge data together. It starts at the deepest levels and merges data tokens together into `DataEntry` items and gradually works its way up to the surface level. Data entries are merged if they share the same unique identifier (i.e. image name, filename, image id, etc.), and bubbled up to the top. Then we can view our data in a Pandas dataframe:

```python
cvdata.dataframe
```

And done! Of course, there are other functions, and a lot of functions behind the scenes to clean up the (potentially) messy data and address inconsistencies. 