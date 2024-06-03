from typing import Union, Any
import xmltodict

from .._utils import load_config
from ..parsing.static import Static
from ..parsing.generic import Generic
from .datafile import DataFile

config = load_config()

class XMLFile(DataFile):
    '''
    The `XMLFile` class represents an annotation object and is similar to the `JSONFile` class
    in terms of hierarchical structure and parsing. The one key difference is the needed usage of
    `AmbiguousList` over `GenericList`, as the presence of multiple tags of the same name will be
    parsed as a list, while tags of one name will be parsed as an item. The algorithm appropriately
    interprets list objects as `AmbiguousList` for this exact reason in `XMLFile`, but if one
    desires a `GenericList` it will have to be instantiated manually.
    
    The structure follows suit to the hierarchy, just as in `JSONFile`. Here is a snippet from the
    Oxford-IIIT Pets Dataset:
    
    .. code-block:: xml
    
        <annotation>
            <folder>OXIIIT</folder>
            <filename>Abyssinian_1.jpg</filename>
            <source>
                <database>OXFORD-IIIT Pet Dataset</database>
                <annotation>OXIIIT</annotation>
                <image>flickr</image>
            </source>
            <size>
                <width>600</width>
                <height>400</height>
                <depth>3</depth>
            </size>
            <segmented>0</segmented>
            <object>
                <name>cat</name>
                <pose>Frontal</pose>
                <truncated>0</truncated>
                <occluded>0</occluded>
                <bndbox>
                    <xmin>333</xmin>
                    <ymin>72</ymin>
                    <xmax>425</xmax>
                    <ymax>158</ymax>
                </bndbox>
                <difficult>0</difficult>
            </object>
        </annotation>
    
    Here we do not specify the extraneous information and get straight to the point:
    
    .. code-block:: python
    
        XMLFile({
            "annotation": {
                "filename": Generic("{}.jpg", DT.IMAGE_NAME),
                "object": AmbiguousList({
                    "name": DT.BBOX_CLASS_NAME,
                    "bndbox": {
                        "xmin": DT.XMIN,
                        "ymin": DT.YMIN,
                        "xmax": DT.XMAX,
                        "ymax": DT.YMAX
                    }
                })
            }
        })
    
    :param form: The form which matches the data to be read from `XMLFile`.
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
            data = xmltodict.parse(f.read())
        return expand_generics(
            curr_path,
            data,
            self.form,
            xml = True
        )
