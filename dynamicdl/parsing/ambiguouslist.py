from typing import Union, Any

from .._utils import union
from .genericlist import GenericList
from .static import Static

class AmbiguousList:
    '''
    Ambiguous List. Used to represent when an item could either be in a list, or a solo item.
    This is primarily used for XML files.
    
    Example:
    
    .. code-block:: xml

        <annotation>
            <box>
                <x1>1.0</x1>
                <x2>2.0</x2>
                <y1>3.0</x1>
                <y2>4.0</y2>
            </box>
        <annotation>
        <annotation>
            <box>
                <x1>1.0</x1>
                <x2>2.0</x2>
                <y1>3.0</x1>
                <y2>4.0</y2>
            </box>
            <box>
                <x1>5.0</x1>
                <x2>6.0</x2>
                <y1>7.0</x1>
                <y2>8.0</y2>
            </box>
        <annotation>
        
    Observe that the above XML file contains potentially multiple `box` tags. When the XML parser
    encounters a tag, it is inferred to be a single tag such that for the first annotation, `box`
    is a dict value with keys `x1`, `x2`, `y1`, `y2` but for the second annotation `box` is a list
    of dicts following the form previously. In this case we wish to use `AmbiguousList` to
    disambiguate the usage of the provided form with an XML file. `AmbiguousList` performs
    identically to `GenericList` for multiple objects, and is primarily separate in order to detect
    otherwise invisible errors with dataset parsing.
    
    :param form: Essentially a wrapper for `GenericList`. Either can provide the args to instantiate
        a `GenericList`, or provide the `GenericList` object itself.
    :type form: GenericList | list | Any
    '''
    def __init__(self, form: Union[GenericList, list, Any]):
        self.form = form if isinstance(form, GenericList) else GenericList(form)

    def expand(
        self,
        path: list[str],
        dataset: Any
    ) -> dict[Static, Any]:
        '''
        Expand potential list into dict of statics.
        
        :param dataset: The dataset data, which is either a single value or a list of values
            following some format.
        :type dataset: Any
        :return: The parsed expansion of `Static` values, always a list. Single values are converted
            to lists of length 1. Note: for consistency lists are converted to dicts with int keys.
        :rtype: dict[int, Any]
        '''
        dataset = union(dataset)
        return self.form.expand(path, dataset)
