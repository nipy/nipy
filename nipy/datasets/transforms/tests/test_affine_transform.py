
"""
This test can only be run from the directory above, as it uses relative
imports.
"""

import numpy as np
import copy
# Don't import from nipy.testing not to have a hard dependence on nipy,
# use np.testing or nose
from nose.tools import assert_equal, assert_true

from ..affine_transform import AffineTransform

def test_compose_with_inverse():
    """ Check that an affine transform composed with its inverse returns
        the identity transform, and the taking the inverse twice gives
        the same transform.
    """
    for _ in range(10):
        affine = np.eye(4)
        affine[:3, :3] = np.random.random((3, 3))
        transform = AffineTransform('in', 'out', affine)
        identity = transform.composed_with(
                        transform.get_inverse())
        yield np.testing.assert_almost_equal, identity.affine, \
                        np.eye(4)
        yield assert_equal, transform, \
                transform.get_inverse().get_inverse()

        x, y, z = np.random.random((3, 10))
        x_, y_, z_ = transform.mapping(*transform.inverse_mapping(x, y, z))
        yield np.testing.assert_almost_equal, x, x_
        yield np.testing.assert_almost_equal, y, y_
        yield np.testing.assert_almost_equal, z, z_


def test_misc():
    """ Test misc private methods for AffineTransform.
    """
    transform = AffineTransform('in', 'out', np.random.random((3, 3)))

    # Check that the repr does not raise an error:
    yield assert_true, isinstance(repr(transform), str)
    # Check that copy and eq work
    yield assert_equal, transform, copy.copy(transform)
    yield assert_equal, transform, copy.deepcopy(transform)

