from typing import Any
from typing_extensions import Self

# from .._warnings import Warnings
from .tokens import RedundantToken
from .datatype import DataType

class PartialType(DataType):
    '''
    The `PartialType` class is used to create datatypes which comprise of another type. This is
    especially useful, for example, when one wishes to use an `IMAGE_ID`-like parameter but it is
    not distinct throughout the dataset. Instead, suppose that the combination of `CLASS_NAME` and
    `IMAGE_ID` forms some unique `IMAGE_NAME`. `PartialType` and its wrapper `ComboType` achieves
    this functionality.
    
    Example:
    
    .. code-block:: python

        my_id_type = DataType('my_id', IDToken())
        my_image_name = ComboType(
            DataTypes.IMAGE_NAME,
            '{}_{}',
            DataTypes.CLASS_NAME,
            my_id_type,
            preserve_all = True
        )
        # ... other code
        # we can now place `my_id_type` and `DataTypes.CLASS_NAME` in our form when parsing the
        # dataset, and when they are found together they will automatically parse into
        # DataTypes.IMAGE_NAME!
    
    Now, every IMAGE_NAME datatype will be constructed from the template `{CLASS_NAME}_{ID}` as we
    have specified. This no longer creates merge conflicts! Note that we created a new ID type that
    is not exactly IMAGE_ID, as IMAGE_ID is a unique token which should not be merged.
    
    :param desc: The purpose of the DataType. This should be unique for every new object.
    :type desc: str
    :param token_type: The token type of the DataType.
    :type token_type: Token
    '''
    def _initialize(self, parent: 'ComboType') -> Self:
        self.parent: ComboType = parent
        return self

class ComboType(DataType):
    '''
    The `ComboType` class is used to create datatypes which comprise of another type. This is
    especially useful, for example, when one wishes to use an `IMAGE_ID`-like parameter but it is
    not distinct throughout the dataset. Instead, suppose that the combination of `CLASS_NAME` and
    `IMAGE_ID` forms some unique `IMAGE_NAME`. `PartialType` and its wrapper `ComboType` achieves
    this functionality. See the `PartialType` class for details.
    
    :param to: The `DataType` to convert the fully initialized `PartialType` collection to.
    :type to: DataType
    :param constructor: The structure to apply when converting to the `DataType`. Each `PartialType`
        section should be replaced with a wildcard `{}` with the order presented in `datatypes`.
    :type constructor: str
    :param datatypes: The PartialTypes for which to make up the ComboType.
    :type datatypes: PartialType
    :param preserve_all: Preserves the data for each `PartialType` in the dataframe; otherwise,
        constructing the `ComboType` will result in popping all `PartialType` data.
    :type preserve_all: bool
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
