# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This test can only be run from the directory above, as it uses relative
imports.
"""

import copy

import numpy as np
import pytest

from ..transform import CompositionError, Transform


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
    pytest.raises(CompositionError, t1.composed_with, t1)

    # Check forward composition (transforms have forward mappings)
    t12 = t1.composed_with(t2)
    x, y, z = np.random.random((3, 10))
    np.testing.assert_equal(mapping(x, y, z), t12.mapping(x, y, z))

    # Check backward composition (transforms have reverse mappings)
    t21 = t2.get_inverse().composed_with(t1.get_inverse())
    x, y, z = np.random.random((3, 10))
    np.testing.assert_equal(mapping(x, y, z), t21.inverse_mapping(x, y, z))

    # Check that you cannot compose transforms that do not have chainable
    # mappings
    pytest.raises(CompositionError, t1.composed_with, \
                        t1.get_inverse())

def test_misc():
    """ Test misc private methods for Transform.
    """
    # Check that passing neither a mapping, nor an inverse_mapping raises
    # a ValueError
    pytest.raises(ValueError, Transform, 'world1', 'world2')

    transform = Transform('in', 'out', mapping=mapping)

    # Check that the repr does not raise an error:
    assert isinstance(repr(transform), str)
    # Check that copy and eq work
    assert transform == copy.copy(transform)


def test_inverse():
    t1 = Transform('in',  'mid', mapping=id, inverse_mapping=id)
    t2 = Transform('mid', 'out', mapping=mapping)
    t3 = Transform('mid', 'out', inverse_mapping=mapping)
    for t in (t1, t2, t3):
        assert t.get_inverse().get_inverse() == t
