# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the discrete_domain utilities.

Caveat assumes that the MNI template image is available at
in ~/.nipy/tests/data
"""

import numpy as np
from numpy.testing import assert_equal

from ..hroi import HROI_as_discrete_domain_blobs, make_hroi_from_subdomain
from ..mroi import subdomain_from_array
from ..discrete_domain import domain_from_array

shape = (5, 6, 7)


def make_domain():
    """Create a multiple ROI instance
    """
    labels = np.ones(shape)
    dom = domain_from_array(labels, affine=None)
    return dom


#######################################################################
# Test on hierarchical ROI
#######################################################################

def make_hroi(empty=False):
    """Create a multiple ROI instance
    """
    labels = np.zeros(shape)
    parents = np.array([])
    if not empty:
        labels[4:, 5:, 6:] = 1
        labels[:2, 0:2, 0:2] = 2
        labels[:2, 5:, 6:] = 3
        labels[:2, 0:2, 6:] = 4
        labels[4:, 0:2, 6:] = 5
        labels[4:, 0:2, 0:2] = 6
        labels[4:, 5:, 0:2] = 7
        labels[:2, 5:, 0:2] = 8
        parents = np.zeros(9)
    else:
        labels = -np.ones(shape)
        parents = np.array([])

    sd = subdomain_from_array(labels, affine=None, nn=0)
    hroi = make_hroi_from_subdomain(sd, parents)
    return hroi


def test_hroi():
    """ Test basic construction of mulitple_roi
    """
    hroi = make_hroi()
    assert_equal(hroi.k, 9)


def test_hroi_isleaf():
    """ Test basic construction of a tree of isolated leaves
    """
    hroi = make_hroi()
    valid = np.ones(9).astype(np.bool)
    valid[1] = 0
    hroi.select(valid)
    assert_equal(hroi.k, 8)


def test_hroi_isleaf_2():
    """Test tree pruning, with parent remapping
    """
    hroi = make_hroi()
    valid = np.ones(9).astype(np.bool)
    valid[0] = 0
    hroi.select(valid)
    assert_equal(hroi.parents, np.arange(8).astype(np.int))


def test_asc_merge():
    """ Test ascending merge
    """
    hroi = make_hroi()
    s1 = hroi.size[0] + hroi.size[1]
    valid = np.ones(9).astype(np.bool)
    valid[1] = 0
    hroi.merge_ascending(valid)
    assert_equal(hroi.size[0], s1)


def test_asc_merge_2():
    """ Test ascending merge
    """
    hroi = make_hroi()
    s1 = hroi.size.copy()
    valid = np.ones(9).astype(np.bool)
    valid[0] = 0
    hroi.merge_ascending(valid)
    assert_equal(hroi.size, s1)


def test_asc_merge_4():
    """Test ascending merge

    """
    hroi = make_hroi()
    hroi.make_feature('labels', hroi.label)
    valid = np.ones(9).astype(np.bool)
    valid[0] = 0
    parents = hroi.parents
    parents = parents + 1
    parents[8] = 8
    hroi.parents = parents
    hroi.merge_ascending(valid)
    assert_equal(hroi.k, 8)
    print hroi.get_feature('labels')[0]
    assert_equal(len(np.asarray(hroi.get_feature('labels')[0]).shape), 1)
    assert_equal(np.unique(hroi.get_feature('labels')[0]).size, 2)
    assert_equal(np.unique(hroi.get_feature('labels')[1]).size, 1)


def test_desc_merge():
    """ Test descending merge
    """
    hroi = make_hroi()
    parents = np.arange(hroi.k)
    parents[1] = 0
    hroi.parents = parents
    s1 = hroi.size[0] + hroi.size[1]
    hroi.merge_descending()
    assert_equal(hroi.size[0], s1)


def test_desc_merge_2():
    """ Test descending merge
    """
    hroi = make_hroi()
    parents = np.maximum(np.arange(-1, hroi.k - 1), 0)
    hroi.parents = parents
    hroi.merge_descending()
    assert_equal(hroi.k, 1)


def test_desc_merge_3():
    """ Test descending merge
    """
    hroi = make_hroi()
    parents = np.minimum(np.arange(1, hroi.k + 1), hroi.k - 1)
    hroi.parents = parents
    hroi.merge_descending()
    assert_equal(hroi.k, 1)


def test_leaves():
    """ Test leaves
    """
    hroi = make_hroi()
    size = hroi.size[1:].copy()
    lroi = hroi.reduce_to_leaves()
    assert_equal(lroi.k, 8)
    print lroi.size, size
    assert_equal(lroi.size, size)


def test_leaves_empty():
    """Test the reduce_to_leaves method on an HROI containing no node

    """
    hroi = make_hroi(empty=True)
    lroi = hroi.reduce_to_leaves()
    assert_equal(lroi.k, 0)


def test_hroi_from_domain():
    dom = make_domain()
    data = np.random.rand(*shape)
    data[:2, 0:2, 0:2] = 2
    rdata = np.reshape(data, (data.size, 1))
    hroi = HROI_as_discrete_domain_blobs(dom, rdata, threshold=1., smin=0)
    assert_equal(hroi.k, 1)

if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
