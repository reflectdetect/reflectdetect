from itertools import tee, islice, zip_longest
from typing import Iterable, Any


def get_next(some_iterable: Iterable[Any], window: int = 1) -> Iterable[Any]:
    """
    Get a value and the next value from an interable
    :param some_iterable:
    :param window: how many next values to get
    :return: an iterable containing the next value and the current value for each step
    """
    items, nexts = tee(some_iterable, 2)
    nexts = islice(nexts, window, None)
    return zip_longest(items, nexts)
