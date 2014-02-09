""" Testing nibcompat module
"""

import numpy as np

from ..nibcompat import get_dataobj, get_affine, get_header

from numpy.testing import assert_array_equal

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


def test_funcs():
    class OldNib:
        def get_header(self):
            return 1
        def get_affine(self):
            return np.eye(4)
        _data = 3
    class NewNib:
        header = 1
        affine = np.eye(4)
        dataobj = 3
    for img in OldNib(), NewNib():
        assert_equal(get_header(img), 1)
        assert_array_equal(get_affine(img), np.eye(4))
        assert_equal(get_dataobj(img), 3)
