# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from nipy.core.reference.slices import bounding_box, \
  zslice, yslice, xslice
from nipy.core.reference.coordinate_map import AffineTransform

from nose.tools import (
    assert_equal,
    assert_true,
    assert_false)
from numpy.testing import (
    assert_array_equal,
    assert_array_almost_equal)
from nipy.testing import (
    parametric)

# Names for a 3D axis set
names = ['xspace','yspace','zspace']

@parametric
def test_bounding_box():
    shape = (10, 14, 16)
    coordmap = AffineTransform.identity(names)
    yield assert_equal(
        bounding_box(coordmap, shape),
        ((0., 9.), (0, 13), (0, 15)))


@parametric
def test_box_slice():
    t = xslice(5, ([0, 9], 10), ([0, 9], 10))
    yield assert_array_almost_equal(t.affine,
                                    [[ 0.,  0.,  5.],
                                     [ 1.,  0.,  0.],
                                     [ 0.,  1.,  0.],
                                     [ 0.,  0.,  1.]])
    t = yslice(4, ([0, 9], 10), ([0, 9], 10))
    yield assert_array_almost_equal(t.affine,
                                    [[ 1.,  0.,  0.],
                                     [ 0.,  0.,  4.],
                                     [ 0.,  1.,  0.],
                                     [ 0.,  0.,  1.]])
    t = zslice(3, ([0, 9], 10), ([0, 9], 10))
    yield assert_array_almost_equal(t.affine,
                                    [[ 1.,  0.,  0.],
                                     [ 0.,  1.,  0.],
                                     [ 0.,  0.,  3.],
                                     [ 0.,  0.,  1.]])

