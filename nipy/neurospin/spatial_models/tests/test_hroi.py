# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the discrete_domain utilities.

Caveat assumes that the MNI template image is available at
in ~/.nipy/tests/data
"""

import numpy as np
from numpy.testing import assert_almost_equal
from nipy.neurospin.spatial_models.hroi import *
from nipy.neurospin.spatial_models.roi_ import mroi_from_array

shape = (5, 6, 7)

def make_hroi():
    """Create a mulmtiple ROI instance
    """
    labels = np.zeros(shape)
    labels[4:,5:,6:] = 1
    labels[:2,:2,:2] = 2
    labels[:2, 5:, 6:] = 3
    labels[:2, :2, 6:] = 4
    labels[4:, :2, 6:] = 5
    labels[4:, :2, :2] = 6
    labels[4:, 5:, :2] = 7
    labels[:2, 5:, :2] = 8
    labels += 1
    parents = np.zeros(9)
    mroi = mroi_from_array(labels, affine=None)
    hroi = NestedROI(mroi.dim, parents, mroi.coord, mroi.local_volume)
    return hroi, labels

def test_hroi():
    """ Test basic construction of mulitple_roi
    """
    hroi,_ = make_hroi()
    assert hroi.k==9

def test_hroi_isleaf():
    """ Test basic construction of a tree of isolated leaves
    """
    hroi,_ = make_hroi()
    valid = np.ones(9).astype(np.bool)
    valid[1] = 0
    hroi.select(valid)
    assert hroi.k==8

def test_hroi_isleaf_2():
    """ Test tree pruning, with prent remapping 
    """
    hroi,_ = make_hroi()
    valid = np.ones(9).astype(np.bool)
    valid[0] = 0
    hroi.select(valid)
    assert (hroi.parents==np.arange(8).astype(np.int)).all()

def test_asc_merge():
    """ Test ascending merge
    """
    hroi,_ = make_hroi()
    s1 = hroi.size[0] + hroi.size[1]
    valid = np.ones(9).astype(np.bool)
    valid[1] = 0
    hroi.merge_ascending(valid)
    assert hroi.size[0]==s1

def test_asc_merge_2():
    """ Test ascending merge
    """
    hroi,_ = make_hroi()
    s1 = hroi.size.copy()
    valid = np.ones(9).astype(np.bool)
    valid[0] = 0
    hroi.merge_ascending(valid)
    assert (hroi.size==s1).all()



if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])





