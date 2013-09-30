# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the discrete_domain utilities.

Caveat assumes that the MNI template image is available at
in ~/.nipy/tests/data

In those tests, we often access some ROI directly by a fixed index
instead of using the utility functions such as get_id() or select_id().

"""

import numpy as np
from numpy.testing import assert_equal

from ..hroi import HROI_as_discrete_domain_blobs, make_hroi_from_subdomain
from ..mroi import subdomain_from_array
from ..discrete_domain import domain_from_binary_array

shape = (5, 6, 7)


def make_domain():
    """Create a multiple ROI instance
    """
    labels = np.ones(shape)
    dom = domain_from_binary_array(labels, affine=None)
    return dom


#######################################################################
# Test on hierarchical ROI
#######################################################################

def make_hroi(empty=False):
    """Create a multiple ROI instance
    """
    labels = np.zeros(shape)
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
    """
    """
    hroi = make_hroi()
    assert_equal(hroi.k, 9)


def test_hroi_isleaf():
    """ Test basic construction of a tree of isolated leaves
    """
    hroi = make_hroi()
    hroi.select_roi([0] + range(2, 9))
    assert_equal(hroi.k, 8)


def test_hroi_isleaf_2():
    """Test tree pruning, with parent remapping
    """
    hroi = make_hroi()
    #import pdb; pdb.set_trace()
    hroi.select_roi(range(1, 9))
    assert_equal(hroi.parents, np.arange(8).astype(np.int))


def test_asc_merge():
    """ Test ascending merge
    """
    hroi = make_hroi()
    s1 = hroi.get_size(0) + hroi.get_size(1)
    total_size = np.sum([hroi.get_size(id) for id in hroi.get_id()])
    assert_equal(hroi.get_size(0, ignore_children=False), total_size)
    hroi.merge_ascending([1])
    assert_equal(hroi.get_size(0), s1)


def test_asc_merge_2():
    """ Test ascending merge

    Test that ROI being their own parent are inchanged.
    """
    hroi = make_hroi()
    s1 = hroi.get_size(0)
    hroi.merge_ascending([0])
    assert_equal(hroi.k, 9)
    assert_equal(hroi.get_size(0), s1)


def test_asc_merge_3():
    """Test ascending merge
    """
    hroi = make_hroi()
    hroi.set_roi_feature('labels', np.arange(9))
    hroi.set_roi_feature('labels2', np.arange(9))
    hroi.merge_ascending([1], pull_features=['labels2'])
    assert_equal(hroi.get_roi_feature('labels', 0), 0)
    assert_equal(hroi.get_roi_feature('labels2', 0), 1)


def test_asc_merge_4():
    """Test ascending merge

    """
    hroi = make_hroi()
    hroi.set_roi_feature('labels', range(9))
    hroi.set_roi_feature('labels2', range(9))
    parents = np.arange(9) - 1
    parents[0] = 0
    hroi.parents = parents
    labels3 = [hroi.label[hroi.label == k] for k in range(hroi.k)]
    hroi.set_feature('labels3', labels3)
    hroi.merge_ascending([1], pull_features=['labels2'])
    assert_equal(hroi.k, 8)
    assert_equal(hroi.get_roi_feature('labels', 0), 0)
    assert_equal(hroi.get_roi_feature('labels2', 0), 1)
    assert_equal(len(hroi.get_feature('labels3')), hroi.k)
    assert_equal(hroi.get_roi_feature('labels2').size, hroi.k)


def test_desc_merge():
    """ Test descending merge
    """
    hroi = make_hroi()
    parents = np.arange(hroi.k)
    parents[1] = 0
    hroi.parents = parents
    s1 = hroi.get_size(0) + hroi.get_size(1)
    hroi.merge_descending()
    assert_equal(hroi.get_size()[0], s1)


def test_desc_merge_2():
    """ Test descending merge
    """
    hroi = make_hroi()
    parents = np.arange(-1, hroi.k - 1)
    parents[0] = 0
    hroi.parents = parents
    hroi.set_roi_feature('labels', np.arange(hroi.k))
    labels2 = [hroi.label[hroi.label == k] for k in range(hroi.k)]
    hroi.set_feature('labels2', labels2)
    hroi.merge_descending()
    assert_equal(hroi.k, 1)
    assert_equal(len(hroi.get_feature('labels2')), hroi.k)
    assert_equal(hroi.get_roi_feature('labels').size, hroi.k)


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
    size = hroi.get_size()[1:].copy()
    lroi = hroi.copy()
    lroi.reduce_to_leaves()
    assert_equal(lroi.k, 8)
    assert_equal(lroi.get_size(), size)
    assert_equal(lroi.get_leaves_id(), np.arange(1, 9))


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


def test_sd_representative():
    """Test the computation of representative features
    """
    hroi = make_hroi()
    hroi.parents = np.arange(9)
    hroi.parents[2] = 1
    data = [[k] * hroi.get_size(k) for k in hroi.get_id()]
    hroi.set_feature('data', data)
    sums = hroi.representative_feature('data')
    for k in hroi.get_id():
        assert_equal(sums[hroi.select_id(k)], k)
    sums2 = hroi.representative_feature('data', ignore_children=False)
    for k in hroi.get_id():
        if k != 1:
            assert_equal(sums2[hroi.select_id(k)], k)
        else:
            assert_equal(sums2[1], 17. / 9)


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
