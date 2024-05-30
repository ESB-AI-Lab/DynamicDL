from abc import ABC, abstractmethod

class DataFile(ABC):
    '''
    Abstract base class for classes that parse annotation files.
    '''
    @abstractmethod
    def parse(
        self,
        path: str,
        curr_path: list[str]
    ) -> dict:
        '''
        Parses a file.
        
         - `path` (`str`): the path to the file.
        '''
