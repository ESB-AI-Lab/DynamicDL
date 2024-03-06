'''
Example file
'''

import json
from src import *

if __name__ == '__main__':
    my_dataset = Dataset(
        ModeToken.classification(),
        PathToken.init_root('/Users/atong/Documents/Datasets'),
        [
            DirectoryToken(
                'annotations',
                Directory.ANNOTATIONS,
                GenericFileToken(StringFormatToken('{}.txt', DataTypes.IMAGE_SET),
                                    File.ANNOTATION)
            ),
            DirectoryToken(
                'images',
                Directory.IMAGES,
                GenericFileToken(
                    StringFormatToken('{}.jpg', PatternAlias(['{}', '{}_{}'],
                        [DataTypes.IMAGE_NAME, [DataTypes.CLASS_NAME, DataTypes.GENERIC]])),
                    File.IMAGE)
            )
        ],
        TXTToken(
            StringFormatToken('{} {} {} {}', [
                PatternAlias(['{}', '{}_{}'],
                             [DataTypes.IMAGE_NAME, [DataTypes.CLASS_NAME, DataTypes.GENERIC]]),
                DataTypes.GENERIC,
                DataTypes.GENERIC,
                DataTypes.CLASS_ID]),
            ignore_type='#'
        )
    )

    print(my_dataset)
    with open('data.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps([str(item) for item in my_dataset.data], indent=4))
