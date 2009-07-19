"""
This test can only be run from the directory above, as it uses relative
imports.
"""

import numpy as np
import copy
# Don't import from nipy.testing not to have a hard dependence on nipy,
# use np.testing or nose
from nose.tools import assert_equal, assert_raises

from ..transform import Transform, CompositionError

################################################################################
# Mappings
def id(x, y, z):
    return x, y, z


def mapping(x, y, z):
    return 2*x, y, 0.5*z

################################################################################
# Tests
def test_composition():
    t1 = Transform('in',  'mid', mapping=id)
    t2 = Transform('mid', 'out', mapping=mapping)
    yield assert_raises, CompositionError, t1.composed_with, t1

    t12 = t1.composed_with(t2)
    x, y, z = np.random.random((3, 10))
    yield np.testing.assert_equal, mapping(x, y, z), \
                t12.mapping(x, y, z)


def test_misc():
    """ Test misc private methods for AffineTransform.
    """
    transform = Transform('in', 'out', mapping=mapping)

    # Check that the repr does not raise an error:
    yield np.testing.assert_, isinstance(repr(transform), str)
    # Check that copy and eq work
    yield assert_equal, transform, copy.copy(transform)

def test_inverse():
    t1 = Transform('in',  'mid', mapping=id, inverse_mapping=id)
    t2 = Transform('mid', 'out', mapping=mapping)
    t3 = Transform('mid', 'out', inverse_mapping=mapping)
    for t in (t1, t2, t3):
        yield assert_equal, t.get_inverse().get_inverse(), t
