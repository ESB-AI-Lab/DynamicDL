from abc import ABC, abstractmethod
from typing import Optional
from tqdm import tqdm

class DataFile(ABC):
    '''
    Abstract base class for classes that parse annotation files.
    '''
    @abstractmethod
    def parse(
        self,
        path: str,
        curr_path: list[str],
        pbar: Optional[tqdm],
        depth: int = 0
    ) -> dict:
        '''
        Parses a file.
        
         - `path` (`str`): the path to the file.
        '''
