'''
Example file

For Coco:

Dataset(
    ModeToken.detection(),
    PathToken.init_root('/Users/atong/Documents/Datasets/Avo')
    [
        DirectoryToken(
            'images',
            Directory.IMAGES,
            GenericFileToken(StringFormatToken('{}.jpg', DataTypes.GENERIC, File.IMAGE))
        ),
        FileToken(
            'result.json',
            File.ANNOTATION
        )
    ],
    JSONToken(
        {
            'images': {
                'id': DataTypes.IMAGE_ID,
                'file_name': StringFormatToken(
                    '{}/{}', [DataTypes.GENERIC, DataTypes.RELATIVE_FILE]
                )
            },
            'categories': {
                'id': DataTypes.CLASS_ID,
                'name': DataTypes.CLASS_NAME
            },
            'annotations': [
                GenericDict({
                    'image_id': DataTypes.IMAGE_ID,
                    'category_id': DataTypes.CLASS_ID,
                    'bbox': [
                        DataTypes.XMIN, DataTypes.YMIN, DataTypes.XMAX, DataTypes.YMAX
                    ]
                })
            ]
        }
    )
)
'''

import json
from src import *

if __name__ == '__main__':
    annotations = TXTToken(
        StringFormatToken('{} {} {} {}', [
            PatternAlias(['{}', '{}_{}'],
                            [DataTypes.IMAGE_NAME, [DataTypes.CLASS_NAME, DataTypes.GENERIC]]),
            DataTypes.GENERIC,
            DataTypes.GENERIC,
            DataTypes.CLASS_ID]),
        ignore_type='#'
    )
    classes = TXTToken(
        StringFormatToken(
            '{} {}', [DataTypes.CLASS_NAME, DataTypes.CLASS_ID]
        )
    )

    my_dataset = Dataset(
        ModeToken.classification(),
        PathToken.init_root('/Users/atong/Documents/Datasets/OxfordPets'),
        [
            DirectoryToken(
                'annotations',
                Directory.ANNOTATIONS,
                GenericFileToken(StringFormatToken('{}.txt', DataTypes.IMAGE_SET),
                                    File.ANNOTATION, format_token=annotations)
            ),
            DirectoryToken(
                'images',
                Directory.IMAGES,
                GenericFileToken(
                    StringFormatToken('{}.jpg', PatternAlias(['{}', '{}_{}'],
                        [DataTypes.IMAGE_NAME, [DataTypes.CLASS_NAME, DataTypes.GENERIC]])),
                    File.IMAGE)
            ),
            FileToken(
                'class.txt',
                purpose=File.ANNOTATION,
                format_token=classes
            )
        ]
    )

    print(my_dataset)
    with open('data.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps([str(item) for item in my_dataset.data], indent=4))
