""" Testing utilities module
"""

import numpy as np

from ..utilities import is_iterable, is_numlike, seq_prod


def test_is_iterable():
    assert is_iterable(())
    assert is_iterable([])
    assert is_iterable(np.zeros(1))
    assert is_iterable(np.zeros((1, 1)))
    assert is_iterable('')
    assert not is_iterable(0)
    assert not is_iterable(object())

    def gen():
        yield 1

    assert is_iterable(gen())

    def func():
        return 1

    assert not is_iterable(func)

    class C:
        def __iter__(self):
            return self

        def __next__(self):
            return self

    assert is_iterable(C())


def test_is_numlike():
    for good in (1, 0, 1.1, False, True, np.zeros(1), np.zeros((3,)),
                 1j, np.complex128(1)):
        assert is_numlike(good)
    for bad in ('', object(), np.array(''), [], [1], (), (1,)):
        assert not is_numlike(bad)


def test_seq_prod():
    assert seq_prod(()) == 1
    assert seq_prod((), 2) == 2
    assert seq_prod((1,)) == 1
    assert seq_prod((1, 2)) == 2
    assert seq_prod((1, 2), 2) == 4
    assert seq_prod((1, 2), 2.) == 4.
