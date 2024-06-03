from typing import Union, Any
import yaml


from .._utils import load_config
from ..parsing.static import Static
from ..parsing.generic import Generic
from .datafile import DataFile

config = load_config()

class YAMLFile(DataFile):
    '''
    The `XMLFile` class represents an annotation object and is similar to the `JSONFile` class
    in terms of hierarchical structure and parsing.
    
    The structure follows suit to the hierarchy, just as in `JSONFile`. Here is a snippet from the
    Tomato Leaf Diseases Dataset:
    
    .. code-block:: yaml
    
        train: ../train/images
        val: ../valid/images
        test: ../test/images

        nc: 7
        names: ['Bacterial Spot', 'Early_Blight', 'Healthy', 'Late_blight', 'Leaf Mold', \
'Target_Spot', 'black spot']

        roboflow:
        workspace: sylhet-agricultural-university
        project: tomato-leaf-diseases-detect
        version: 3
        license: Public Domain
    
    Of particular interest is the `names` list, in which we need an `ImpliedList` to set up a
    pairing between class ID and class name. We do exactly that:
    
    .. code-block:: python
    
        YAMLFile({
            'names': Pairing(
                ImpliedList([DT.BBOX_CLASS_NAME], indexer=DT.BBOX_CLASS_ID),
                DT.BBOX_CLASS_NAME, DT.BBOX_CLASS_ID
            )
        })
    
    :param form: The form which matches the data to be read from JSONFile.
    :type form: dict[str | DataType | Static | Generic | Alias, Any] | list[Any]
    '''
    def __init__(self, form: dict[Union[Static, Generic], Any]) -> None:
        self.form = form

    def parse(
        self,
        path: str,
        curr_path: list[str]
    ) -> dict:
        from .._main._engine import expand_generics
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return expand_generics(
            curr_path,
            data,
            self.form
        )
