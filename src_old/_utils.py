def union(item: list[object] | object) -> list[object]:
    '''
    Returns the item in a list format if not already in a list format.
    '''
    if not isinstance(item, list):
        item = [ item ]
    return item
