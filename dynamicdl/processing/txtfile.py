
from typing import Union, Optional, Any
from tqdm import tqdm

from .._utils import load_config, union
from ..data.datatype import DataType
from ..data.datatypes import DataTypes
from ..parsing.static import Static
from ..parsing.generic import Generic
from .datafile import DataFile
from ..parsing.pairing import Pairing

config = load_config()

class TXTFile(DataFile):
    '''
    Utility functions for parsing txt files.
    '''
    def __init__(
        self,
        form: Union[dict, list],
        ignore_type: Optional[Union[list[Union[Generic, str]], Generic, str]] = None
    ) -> None:
        self.form = form
        self.named = isinstance(form, dict)
        self.ignore_type: list[Generic] = []
        if ignore_type:
            ignore_type = union(ignore_type)
            self.ignore_type = [Generic(rule + '{}', DataTypes.GENERIC) if
                           isinstance(rule, str) else rule for rule in ignore_type]

    def parse(
        self,
        path: str,
        curr_path: list[str],
        pbar: Optional[tqdm],
        depth: int = 0
    ) -> dict:
        from .._main._engine import expand_generics
        def filter_ignores(line: str):
            for ignore_type in self.ignore_type:
                if ignore_type.match(line)[0]:
                    return True
            return False
        if depth >= config['MAX_PBAR_DEPTH']:
            pbar = None
        if pbar:
            pbar.set_description(f'Expanding generics: {"/".join(curr_path)}')
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if filter_ignores(line):
                    continue
                data.append(line)
        data, _ = TXTFile._parse(data, self.form)
        return expand_generics(
            curr_path,
            data,
            self.form,
            pbar,
            depth = depth
        )

    @staticmethod
    def _parse(data: list[str], form: Any) -> Any:
        if isinstance(form, Pairing):
            form = form.form
        if isinstance(form, (Generic, Static, str, DataType)):
            if isinstance(form, str):
                form = Static(form)
            elif isinstance(form, DataType):
                form = Generic('{}', form)
            if form.match(data[0]):
                return data, 1
            raise ValueError("TXTFile Failed to parse")
        if isinstance(form, dict):
            return TXTFile._parse_dict(data, form)
        if isinstance(form, list):
            return TXTFile._parse_list(data, form)
        raise ValueError("Unknown Token")

    @staticmethod
    def _parse_list(data: list[str], form: list) -> list:
        parsed_data = []
        ctr = 0
        i = 0
        while True:
            if ctr >= len(data):
                return parsed_data, ctr
            next_form = form[i]
            if isinstance(next_form, (list, dict)):
                obj_data, endline = TXTFile._parse(data[ctr:], form)
                parsed_data.append(obj_data)
                ctr += endline
                i = (i + 1) % len(form)
                continue
            if isinstance(next_form, str):
                next_form = Static(next_form)
            elif isinstance(next_form, DataType):
                next_form = Generic('{}', next_form)
            result, _ = next_form.match(data[ctr])
            if not result:
                return parsed_data, ctr
            parsed_data.append(data[ctr])
            ctr += 1
            i = (i + 1) % len(form)

    @staticmethod
    def _parse_dict(data: list[str], form: dict) -> dict:
        cleaned_form: dict[Union[Static, Generic], Any] = {}
        for generic, subform in form.items():
            if isinstance(generic, str):
                generic = Static(generic)
            elif isinstance(generic, DataType):
                generic = Generic('{}', generic)
            cleaned_form[generic] = subform
        form: dict[Union[Static, Generic], Any] = cleaned_form
        parsed_data = {}
        prev = -1
        start = -1
        for i, line in enumerate(data):
            result = False
            for generic in form:
                result, _ = generic.match(line)
                if result:
                    break
            if not result:
                continue
            if start != -1:
                parsed_data[prev_line], _ = TXTFile._parse(data[start:i], form[key])
            prev = start
            start = i + 1
            key = generic
            prev_line = line
        if start != prev:
            parsed_data[prev_line], start = TXTFile._parse(data[start:], form[key])
        else:
            start = 0
        return parsed_data, start
