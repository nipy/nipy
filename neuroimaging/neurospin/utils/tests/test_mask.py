"""
Test the mask-extracting utilities.
"""

from nipy.neurospin.utils.mask import _largest_cc
from numpy.testing import assert_equal
import numpy as np


def test_largest_cc():
    """ Check the extraction of the largest connected component.
    """
    a = np.zeros((6, 6, 6))
    a[1:3, 1:3, 1:3] = 1
    assert_equal(a, _largest_cc(a))
    b = a.copy()
    b[5, 5, 5] = 1
    assert_equal(a, _largest_cc(b))


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])

