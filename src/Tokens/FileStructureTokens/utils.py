'''
Utility functions for structure tokens.
'''
from .StructureTokens import StructureToken
from .GenericTokens import GenericStructureToken

def instantiate_all(structures: list[StructureToken]) -> None:
    '''
    Instantiate a list of structures, including expanding the generic structures.
    Structures is modified in place.
    
    - structures (list[StructureToken]): a list of the structures to instantiate.
    '''
    expanded_structures = []
    for structure in structures:
        if isinstance(structure, GenericStructureToken):
            structure.instantiate()
            expanded_structures += structure.expand()

    # modify in place
    structures += expanded_structures

    for structure in structures:
        structure.instantiate()

    return structures
