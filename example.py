from src import *

if __name__ == '__main__':
    my_dataset = Dataset(
        ModeToken.classification(),
        PathToken.init_root('/Users/atong/Documents/Datasets'),
        [
            DirectoryToken(
                'annotations',
                Directory.GENERIC,
                GenericDirectoryToken(
                    StringFormatToken('{}', DataTypes.IMAGE_SET),
                    Directory.ANNOTATIONS,
                    GenericFileToken(StringFormatToken('{}.txt', DataTypes.IMAGE_SET),
                                     File.ANNOTATION)
                )
            ),
            DirectoryToken(
                'images',
                Directory.IMAGES,
                GenericFileToken(StringFormatToken('{}.jpg', DataTypes.IMAGE_NAME), File.IMAGE)
            )
        ],
        TXTToken(
            StringFormatToken('{} {} {} {}', [DataTypes.IMAGE_NAME, DataTypes.GENERIC,
                              DataTypes.GENERIC, DataTypes.CLASS_ID]), ignore_type='#'
        )
    )
    print(my_dataset)
