from typing import Optional, Union, Any
from tqdm import tqdm
import xmltodict

from .._utils import load_config
from ..parsing.static import Static
from ..parsing.generic import Generic
from .datafile import DataFile

config = load_config()

class XMLFile(DataFile):
    '''
    Utility functions for parsing xml files.
    '''
    def __init__(self, form: dict[Union[Static, Generic], Any]) -> None:
        self.form = form

    def parse(
        self,
        path: str,
        curr_path: list[str],
        pbar: Optional[tqdm],
        depth: int = 0
    ) -> dict:
        from ..engine import expand_generics
        if depth >= config['MAX_PBAR_DEPTH']:
            pbar = None
        if pbar:
            pbar.set_description(f'Expanding generics: {"/".join(curr_path)}')
        with open(path, 'r', encoding='utf-8') as f:
            data = xmltodict.parse(f.read())
        return expand_generics(
            curr_path,
            data,
            self.form,
            pbar,
            depth = depth
        )