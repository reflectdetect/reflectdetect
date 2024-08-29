from itertools import tee, islice, zip_longest
from typing import Iterable, Any


def get_next(some_iterable: Iterable[Any], window: int = 1) -> Iterable[Any]:
    items, nexts = tee(some_iterable, 2)
    nexts = islice(nexts, window, None)
    return zip_longest(items, nexts)
