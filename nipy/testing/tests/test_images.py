# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
''' Test example images '''

from numpy.testing import assert_array_equal

from nipy import load_image
from nipy.testing import funcfile


def test_dims():
    fimg = load_image(funcfile)
    # make sure time dimension is correctly set in affine
    assert_array_equal(fimg.coordmap.affine[3, 3], 2.0)
    # should follow, but also make sure affine is invertible
    ainv = fimg.coordmap.inverse
    assert not ainv is None
