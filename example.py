from src import *

if __name__ == '__main__':
    my_dataset = Dataset(
        ModeToken.classification(),
        PathToken.init_root('/Users/atong/Documents/Datasets'),
        [
            DirectoryToken(
                'annotations',
                DirectoryPurposeToken(),
                GenericDirectoryToken(
                    StringFormatToken(
                        '{}',
                        DataTypes.IMAGE_SET
                    ),
                    DirectoryPurposeToken(),
                    GenericFileToken(
                        StringFormatToken(
                            '{}.txt',
                            DataTypes.IMAGE_SET
                        ),
                        FilePurposeToken()
                    )
                )
            ),
            DirectoryToken(
                'images',
                DirectoryPurposeToken(),
                GenericFileToken(
                    StringFormatToken(
                        '{}.jpg',
                        DataTypes.IMAGE_NAME
                    ),
                    FilePurposeToken()
                )
            )
        ],
        TXTToken(
            
        )
    )
    print(my_dataset)