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
                        Token()
                    ),
                    DirectoryPurposeToken(),
                    GenericFileToken(
                        StringFormatToken(
                            '{}.txt',
                            Token()
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
                        Token()
                    ),
                    FilePurposeToken()
                )
            )
        ],
        TXTToken(

        )
    )
