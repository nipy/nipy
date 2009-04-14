#!/usr/bin/env python

#
# Test BLAS 2
#

from numpy.testing import assert_almost_equal, assert_equal
import numpy as np
import nipy.neurospin.bindings as fb


def test_global():
    'Warning: No blas-2 test implemented yet.'


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])

