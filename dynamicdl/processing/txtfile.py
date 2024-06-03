
from typing import Union, Optional, Any

from .._utils import load_config, union
from ..data.datatype import DataType
from ..data.datatypes import DataTypes
from ..parsing.static import Static
from ..parsing.generic import Generic
from ..parsing.alias import Alias
from .datafile import DataFile
from ..parsing.pairing import Pairing

config = load_config()

class TXTFile(DataFile):
    '''
    The `TXTFile` class is an annotation object notator specifically for `.txt` file parsing. It
    also can parse anything that is represented in plaintext, i.e. with UTF-8 encoding. It takes
    a form similar to any nested dict structure, but it is also dangerous and should be noted
    that distinct lines must take distinct forms for differentiation and disambiguation.
    
    An example of a txt file that we want to parse:
    
    .. code-block:: 
    
        imageset1
        class1
        image1
        1.0 2.0 3.0 4.0
        5.0 6.0 7.0 8.0
        image2
        2.0 3.0 5.6 2.43
        image3
        5.4 12.4 543.2 12.3
        2.0 3.0 5.6 2.44
        2.0 3.0 5.6 2.46
        2.0 3.0 5.6 2.48
        class2
        image4
        32.54 21.4 32.43 12.23
        image5
        imageset2
        class1
        image6
        32.54 21.4 32.43 12.256

        classes
        class1 abc
        class2 def
        class3 ghi
        
    Observe that each line can be distinctly classified in a hierarchical sense. That is, each
    individual line can be attributed to a single purpose.
    
    .. code-block:: python
    
        TXTFile({
            Generic('imageset{}', DT.IMAGE_SET_ID): {
                Generic('class{}', DT.CLASS_ID): {
                    Generic('image{}', DT.IMAGE_ID): [
                        Generic('{} {} {} {}', DT.X1, DT.X2, DT.Y1, DT.Y2)
                    ]
                }
            },
            'classes': Pairing([
                Generic('class{} {}', DT.CLASS_ID, DT.CLASS_NAME)
            ], DT.CLASS_ID, DT.CLASS_NAME)
        })
        
    Notice the natural structure which is inherited. Each generic ends up distinct from each other,
    so the dataset is not ambiguous. A hierarchical structure would look as follows:
    
    .. code-block:: 
    
        imageset1
            class1
                image1
                    1.0 2.0 3.0 4.0
                    5.0 6.0 7.0 8.0
                image2
                    2.0 3.0 5.6 2.43
                image3
                    5.4 12.4 543.2 12.3
                    2.0 3.0 5.6 2.44
                    2.0 3.0 5.6 2.46
                    2.0 3.0 5.6 2.48
            class2
                image4
                    32.54 21.4 32.43 12.23
                image5
        imageset2
            class1
                image6
                    32.54 21.4 32.43 12.256
        classes
            class1 abc
            class2 def
            class3 ghi
        
    Notice that this is exactly the structure reflected in the above code used to parse the file.
    We can also specify an `ignore_type` such that any line which matches the Generic or string
    passed in is skipped.
    
    :param form: The form which matches the data to be read from `TXTFile`.
    :type form: dict[str | DataType | Static | Generic | Alias, Any] | list[Any]
    :param ignore_type: A list, or one value of Generic/str objects which if matched will ignore
        the line parsed.
    :type ignore_type: Optional[Union[list[Union[Generic, str]], Generic, str]]
    '''
    def __init__(
        self,
        form: Union[dict[Union[str, DataType, Static, Generic, Alias], Any], list[Any]],
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
        curr_path: list[str]
    ) -> dict:
        from .._main._engine import expand_generics
        def filter_ignores(line: str):
            for ignore_type in self.ignore_type:
                if ignore_type.match(line)[0]:
                    return True
            return False
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
            self.form
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
