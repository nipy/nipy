from ..histogram import histogram

import numpy as np
from numpy.testing import assert_array_equal


def test_histogram():
    x = np.array([0,
                  1, 1,
                  2, 2, 2,
                  3, 3, 3, 3,
                  4, 4, 4, 4, 4],
                 dtype='uintp')
    h = histogram(x)
    assert_array_equal(h, [1, 2, 3, 4, 5])
