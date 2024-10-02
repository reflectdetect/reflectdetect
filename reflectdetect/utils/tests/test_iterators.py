import pytest
from itertools import tee, islice, zip_longest
from reflectdetect.utils.iterators import get_next


def test_get_next():
    # Test with default window size (1)
    iterable = [1, 2, 3, 4]
    result = list(get_next(iterable))
    assert result == [(1, 2), (2, 3), (3, 4), (4, None)], "Failed with window size 1"

    # Test with custom window size (2)
    result = list(get_next(iterable, window=2))
    assert result == [(1, 3), (2, 4), (3, None), (4, None)], "Failed with window size 2"

    # Test with an empty iterable
    iterable = []
    result = list(get_next(iterable))
    assert result == [], "Failed with empty iterable"

    # Test with iterable containing one element
    iterable = [1]
    result = list(get_next(iterable))
    assert result == [(1, None)], "Failed with one-element iterable"

    # Test with window larger than iterable
    iterable = [1, 2]
    result = list(get_next(iterable, window=3))
    assert result == [(1, None), (2, None)], "Failed with window larger than iterable"

    # Test with string iterable (as strings are also iterable in Python)
    iterable = "abc"
    result = list(get_next(iterable))
    assert result == [('a', 'b'), ('b', 'c'), ('c', None)], "Failed with string iterable"
