from typing import Any
from typing_extensions import Self

# from .._warnings import Warnings
from .tokens import RedundantToken
from .datatype import DataType

class PartialType(DataType):
    '''
    The `PartialType class
    '''
    def _initialize(self, parent: 'ComboType') -> Self:
        self.parent: ComboType = parent
        return self

class ComboType(DataType):
    '''
    The `ComboType` class
    '''
    def __init__(
        self,
        to: DataType,
        constructor: str,
        *datatypes: PartialType,
        preserve_all: bool = False
    ):
        self.datatypes = {dt.desc: dt._initialize(self) for dt in datatypes}
        self.constructor = constructor
        # assert lengths equal
        self.to = to
        self.preserve_all = preserve_all
        super().__init__(
            desc = str(id(self)),
            token_type = to.token_type,
            doc = (f'Combo data type for {to.desc}, comprised of elements '
                f'{", ".join(self.datatypes.keys())}. ' + to.doc)
        )

    def construct(self, values: list[Any]) -> Any:
        '''
        Construct the full datatype value
        '''
        if isinstance(self.token_type, RedundantToken):
            return self.to.token_type.transform(
                [self.constructor.format(*entry) for entry in zip(*values)]
            )
        return self.to.token_type.transform(
            self.constructor.format(*values)
        )
