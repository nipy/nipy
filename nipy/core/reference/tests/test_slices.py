# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from numpy.testing import assert_array_almost_equal

from ..coordinate_map import AffineTransform
from ..coordinate_system import CoordinateSystem as CS
from ..slices import bounding_box, xslice, yslice, zslice
from ..spaces import mni_csm, scanner_csm, scanner_space

# Names for a 3D axis set
names = ['xspace', 'yspace', 'zspace']

def test_bounding_box():
    shape = (10, 14, 16)
    coordmap = AffineTransform.identity(names)
    assert (bounding_box(coordmap, shape) ==
                 ((0., 9.), (0, 13), (0, 15)))


def test_box_slice():
    t = xslice(5, ([0, 9], 10), ([0, 9], 10), scanner_space)
    assert_array_almost_equal(t.affine,
                              [[ 0.,  0.,  5.],
                               [ 1.,  0.,  0.],
                               [ 0.,  1.,  0.],
                               [ 0.,  0.,  1.]])
    assert t.function_domain == CS(['i_y', 'i_z'], 'slice')
    assert t.function_range == scanner_csm(3)
    t = yslice(4, ([0, 9], 10), ([0, 9], 10), 'mni')
    assert_array_almost_equal(t.affine,
                              [[ 1.,  0.,  0.],
                               [ 0.,  0.,  4.],
                               [ 0.,  1.,  0.],
                               [ 0.,  0.,  1.]])
    assert t.function_domain == CS(['i_x', 'i_z'], 'slice')
    assert t.function_range == mni_csm(3)
    t = zslice(3, ([0, 9], 10), ([0, 9], 10), mni_csm(3))
    assert_array_almost_equal(t.affine,
                              [[ 1.,  0.,  0.],
                               [ 0.,  1.,  0.],
                               [ 0.,  0.,  3.],
                               [ 0.,  0.,  1.]])
    assert t.function_domain == CS(['i_x', 'i_y'], 'slice')
    assert t.function_range == mni_csm(3)
