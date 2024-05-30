from typing import Union, Iterable
import csv

from .._utils import load_config
from .._warnings import Warnings
from ..data.datatype import DataType
from ..data.dataentry import DataEntry
from ..parsing.static import Static
from ..parsing.generic import Generic
from ..parsing.alias import Alias
from .datafile import DataFile

config = load_config()

class CSVFile(DataFile):
    '''
    Utility functions for parsing csv files.
    '''
    def __init__(
        self,
        form: Iterable[Union[DataType, Static, Generic, Alias]],
        header: bool = True
    ) -> None:
        self.form = []
        for generic in form:
            if isinstance(generic, DataType):
                self.form.append(Generic('{}', generic))
            else:
                self.form.append(generic)
        self.header = header

    def parse(
        self,
        path: str,
        curr_path: list[str]
    ) -> dict:
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            ncol = len(next(reader))
            if ncol != len(self.form):
                Warnings.error('invalid_csv_cols', n=ncol, exp=len(self.form))
            if not self.header:
                f.seek(0)
            for row in reader:
                items = []
                for item, form in zip(row, self.form):
                    res, data = form.match(item)
                    if not res:
                        Warnings.error('invalid_csv_value', item=item, form=form)
                    items.append(data)
                data.append(DataEntry(items))
        return data
