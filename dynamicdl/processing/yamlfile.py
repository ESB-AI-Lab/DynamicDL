from typing import Union, Any
import yaml


from .._utils import load_config
from ..parsing.static import Static
from ..parsing.generic import Generic
from .datafile import DataFile

config = load_config()

class YAMLFile(DataFile):
    '''
    Utility functions for parsing yaml files.
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
