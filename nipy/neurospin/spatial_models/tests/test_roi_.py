# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the discrete_domain utilities.

Caveat assumes that the MNI template image is available at
in ~/.nipy/tests/data
"""

import numpy as np
from numpy.testing import assert_almost_equal
from nipy.neurospin.spatial_models.roi_ import *
from nipy.io.imageformats import Nifti1Image

shape = (5, 6, 7)

def test_mroi():
    labels = np.zeros(shape)
    labels[4:,5:,6:] = 1
    labels[:2,:2,:2] = 2
    labels[:2, 5:, 6:] = 3
    labels[:2, :2, 6:] = 4
    labels[4:, :2, 6:] = 5
    labels[4:, :2, :2] = 6
    labels[4:, 5:, :2] = 7
    labels[:2, 5:, :2] = 8
    mroi = mroi_from_array(labels, affine=None)
    assert mroi.k==8

if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])





