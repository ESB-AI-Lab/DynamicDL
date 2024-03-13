'''
Example file

For Coco:

Pets:
alias = Alias([
    Generic("{}", DataTypes.IMAGE_NAME),
    Generic("{}_{}", DataTypes.CLASS_NAME, DataTypes.GENERIC)
])
{
    Static("annotations"): Directory({
        Generic("{}.txt", DataTypes.IMAGE_SET): TXTFile([
            Generic("{} {} {} {}", [alias, DataTypes.GENERIC, DataTypes.GENERIC, DataTypes.CLASS_ID)])
        ])
    }),
    Static("images"): Directory({
        Generic("{}.jpg", alias): Image()
    })
}

{
    result.json (STATIC): {
        images (STATIC): [
            {
                width (STATIC): (GENERIC)
                height (STATIC): (GENERIC)
                id (STATIC): (IMAGE_ID)
                file_name (STATIC): (RELATIVE_FILE)
            }
            ...
        ]
        categories (STATIC): [
            {
                id (STATIC): (CLASS_ID)
                name (STATIC): (CLASS_NAME)
            }
            ...
        ]
        annotations (STATIC): [
            {
                id (STATIC): (GENERIC)
                image_id (STATIC): (IMAGE_ID)
                category_id (STATIC): (CLASS_ID)
                bbox (STATIC): [
                    (XMIN),
                    (YMIN),
                    (XMAX),
                    (YMAX)
                ]
            }
            ...
        ]
    }
    images (STATIC): {
        <generic_filename> (ABSOLUTE_FILE, RELATIVE_FILE): (IMAGE)
        ...
    }
}

alias = Alias([
    Generic("{}", DataTypes.IMAGE_NAME),
    Generic("{}_{}", DataTypes.CLASS_NAME, DataTypes.GENERIC)
])
root = '/Users/atong/Documents/Datasets/OxfordPets'
form = {
    Static("annotations"): {
        Generic("{}.txt", DataTypes.IMAGE_SET): TXTFile(
            Generic(
                "{} {} {} {}", alias, DataTypes.GENERIC, DataTypes.GENERIC, DataTypes.CLASS_ID
            ),
            ignore_type = '#'
        )
    },
    Static("images"): {
        Generic("{}.jpg", alias): Image()
    }
}
dataset = Dataset(root, form)
'''

from pprint import pprint
from src import *

if __name__ == '__main__':
    form = {
        Static('images'): {
            Generic("{}.jpg", DataTypes.GENERIC): Image()
        },
        Static('result.json'): JSONFile({
            Static('images'): GenericList([{
                Static('id'): DataTypes.IMAGE_ID,
                Static('file_name'): DataTypes.RELATIVE_FILE
            }]),
            Static('categories'): GenericList([{
                Static('id'): DataTypes.CLASS_ID,
                Static('name'): DataTypes.CLASS_NAME
            }]),
            Static('annotations'): GenericList([{
                Static('image_id'): DataTypes.IMAGE_ID,
                Static('category_id'): DataTypes.CLASS_ID,
                Static('bbox'): GenericList([
                    DataTypes.XMIN, DataTypes.YMIN, DataTypes.XMAX, DataTypes.YMAX
                ])
            }])
        })
    }
    dataset = Dataset('/Users/atong/Documents/Datasets/Avo', form)
    pprint(dataset.dataset)
