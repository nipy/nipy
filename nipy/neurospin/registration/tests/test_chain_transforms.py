""" Testing combined transformation objects

The combined transform object associates a spatial transformation with the
parameters of that transformation, for use in an optimizer.

The combined transform object does several things.  First, it can transform a
coordinate array with::

    transformed_pts = obj.apply(pts)

Second, the transform can phrase itself as a vector of parameters that are
suitable for optimization::

    vec = obj.get_params()

Third, the transform can be modified by setting from the optimization
parameters::

    obj.set_params(new_vec)
    new_transformed_pts = obj.apply(pts)

"""

import numpy as np
import numpy.linalg as npl

from ..chain_transform import ChainTransform
from ..affine import Affine, apply_affine

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_true, assert_equal, assert_raises


def test_creation():
    aff1 = np.diag([2, 3, 4, 1])
    aff2 = np.eye(4)
    aff2[:3,3] = (10, 11, 12)
    # generate a random affine with a positive determinant
    aff3 = np.eye(4)
    aff3[:3,3] = np.random.normal(size=(3,))
    tmp = np.random.normal(size=(3,3))
    aff3[:3,:3] = np.sign(npl.det(tmp))*tmp 

    # Make affine objects
    aff1_obj, aff2_obj, aff3_obj = [Affine(a) for a in [aff1, aff2.copy(), aff3]]
    pts = np.arange(12).reshape(4,3)
    # This is the simplest possible example, where there is a thing we are
    # optimizing, and an optional pre and post transform
    ct = ChainTransform(aff2_obj)
    # Check apply gives expected result
    assert_array_equal(ct.apply(pts),
                       apply_affine(aff2, pts))
    # Check that result is changed by setting params
    assert_array_equal(ct.param, aff2_obj.param)
    ct.param = np.zeros((12,))
    assert_array_equal(ct.apply(pts), pts)
    # Does changing params in chain object change components passed in?
    assert_array_equal(aff2_obj.param, np.zeros((12,)))
    # Reset the aff2 object
    aff2_obj = Affine(aff2.copy())
    # Check apply gives the expected results
    ct = ChainTransform(aff2_obj, pre=aff1_obj)
    assert_array_almost_equal(aff1_obj.as_affine(), aff1)
    assert_array_almost_equal(aff2_obj.as_affine(), aff2)
    tmp = np.dot(aff2, aff1)
    assert_array_almost_equal(ct.apply(pts),
                       apply_affine(np.dot(aff2, aff1), pts))
    # Check that result is changed by setting params
    assert_array_equal(ct.param, aff2_obj.param)
    ct.param = np.zeros((12,))
    assert_array_almost_equal(ct.apply(pts), apply_affine(aff1, pts))
    # Does changing params in chain object change components passed in?
    assert_array_equal(aff2_obj.param, np.zeros((12,)))
    # Reset the aff2 object
    aff2_obj = Affine(aff2.copy())
    ct = ChainTransform(aff2_obj, pre=aff1_obj, post=aff3_obj)
    assert_array_almost_equal(ct.apply(pts),
                       apply_affine(np.dot(aff3, np.dot(aff2, aff1)), pts))
    # Check that result is changed by setting params
    assert_array_equal(ct.param, aff2_obj.param)
    ct.param = np.zeros((12,))
    assert_array_almost_equal(ct.apply(pts), apply_affine(np.dot(aff3, aff1), pts))
    # Does changing params in chain object change components passed in?
    assert_array_equal(aff2_obj.param, np.zeros((12,)))

