import json
from typing import Union, Any

from .._utils import load_config
from ..data.datatype import DataType
from ..parsing.static import Static
from ..parsing.generic import Generic
from ..parsing.alias import Alias
from .datafile import DataFile

config = load_config()

class JSONFile(DataFile):
    '''
    The `JSONFile` class represents an annotation object and has the simplest conversion from the
    form to parsing. Data essentially follows the dict/list format in Python.
    
    Example:
    
    .. code-block:: json

        {
            "images": [
                {
                    "id": 0,
                    "file_name": "sample.jpg"
                }
            ],
            "categories": [
                {
                    "id": 0,
                    "name": "my_class"
                }
            ],
            "annotations": [
                {
                    "image_id": 0,
                    "category_id": 0,
                    "bbox": [1.0, 2.0, 3.0, 4.0]
                }
            ]
        }
    
    .. code-block:: python
    
        JSONFile({
            'images': [{
                'id': DT.IMAGE_ID,
                'file_name': Generic('{}.jpg', DT.IMAGE_NAME)
            }],
            'categories': Pairing([{
                'id': DT.BBOX_CLASS_ID,
                'name': DT.BBOX_CLASS_NAME
            }], DT.BBOX_CLASS_ID, DT.BBOX_CLASS_NAME),
            'annotations': [{
                'image_id': DT.IMAGE_ID,
                'category_id': DT.BBOX_CLASS_ID,
                'bbox': [DT.XMIN, DT.YMIN, DT.WIDTH, DT.HEIGHT]
            }]
        })
    
    Notice how the JSONFile constructor matches exactly the style of the json data, denoting areas
    which can represent data items respectively.
    
    :param form: The form which matches the data to be read from JSONFile.
    :type form: dict[str | DataType | Static | Generic | Alias, Any] | list[Any]
    '''
    def __init__(
        self,
        form: Union[dict[Union[str, DataType, Static, Generic, Alias], Any], list[Any]]
    ) -> None:
        self.form = form

    def parse(
        self,
        path: str,
        curr_path: list[str]
    ) -> dict:
        from .._main._engine import expand_generics
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return expand_generics(
            curr_path,
            data,
            self.form
        )
