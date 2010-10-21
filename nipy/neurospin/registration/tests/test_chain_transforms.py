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

AFF1 = np.diag([2, 3, 4, 1])
AFF2 = np.eye(4)
AFF2[:3,3] = (10, 11, 12)
# generate a random affine with a positive determinant
AFF3 = np.eye(4)
AFF3[:3,3] = np.random.normal(size=(3,))
tmp = np.random.normal(size=(3,3))
AFF3[:3,:3] = np.sign(npl.det(tmp))*tmp 


def test_creation():
    # Make affine objects
    aff1_obj, aff2_obj, aff3_obj = [Affine(a) for a in [AFF1, AFF2.copy(), AFF3]]
    pts = np.arange(12).reshape(4,3)
    # This is the simplest possible example, where there is a thing we are
    # optimizing, and an optional pre and post transform
    ct = ChainTransform(aff2_obj)
    # Check apply gives expected result
    assert_array_equal(ct.apply(pts),
                       apply_affine(AFF2, pts))
    # Check that result is changed by setting params
    assert_array_equal(ct.param, aff2_obj.param)
    ct.param = np.zeros((12,))
    assert_array_equal(ct.apply(pts), pts)
    # Does changing params in chain object change components passed in?
    assert_array_equal(aff2_obj.param, np.zeros((12,)))
    # Reset the aff2 object
    aff2_obj = Affine(AFF2.copy())
    # Check apply gives the expected results
    ct = ChainTransform(aff2_obj, pre=aff1_obj)
    assert_array_almost_equal(aff1_obj.as_affine(), AFF1)
    assert_array_almost_equal(aff2_obj.as_affine(), AFF2)
    tmp = np.dot(AFF2, AFF1)
    assert_array_almost_equal(ct.apply(pts),
                       apply_affine(np.dot(AFF2, AFF1), pts))
    # Check that result is changed by setting params
    assert_array_equal(ct.param, aff2_obj.param)
    ct.param = np.zeros((12,))
    assert_array_almost_equal(ct.apply(pts), apply_affine(AFF1, pts))
    # Does changing params in chain object change components passed in?
    assert_array_equal(aff2_obj.param, np.zeros((12,)))
    # Reset the aff2 object
    aff2_obj = Affine(AFF2.copy())
    ct = ChainTransform(aff2_obj, pre=aff1_obj, post=aff3_obj)
    assert_array_almost_equal(ct.apply(pts),
                       apply_affine(np.dot(AFF3, np.dot(AFF2, AFF1)), pts))
    # Check that result is changed by setting params
    assert_array_equal(ct.param, aff2_obj.param)
    ct.param = np.zeros((12,))
    assert_array_almost_equal(ct.apply(pts), apply_affine(np.dot(AFF3, AFF1), pts))
    # Does changing params in chain object change components passed in?
    assert_array_equal(aff2_obj.param, np.zeros((12,)))


def test_inputs():
    # Check that we can pass arrays or None as pre and post
    ct = ChainTransform(AFF2.copy())
    ct = ChainTransform(AFF2.copy(), pre=AFF1)
    ct = ChainTransform(AFF2.copy(), pre=AFF1, post=AFF3)

