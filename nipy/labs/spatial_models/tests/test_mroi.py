# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the discrete_domain utilities.

Caveat assumes that the MNI template image is available at
in ~/.nipy/tests/data
"""

import numpy as np
from numpy.testing import assert_equal
from ..mroi import *
from ..discrete_domain import domain_from_binary_array

shape = (5, 6, 7)


###########################################################
# SubDomains tests
###########################################################

def make_subdomain():
    """Create a multiple ROI instance
    """
    labels = np.zeros(shape)
    labels[4:, 5:, 6:] = 1
    labels[:2, 0:2, 0:2] = 2
    labels[:2, 5:, 6:] = 3
    labels[:2, 0:2, 6:] = 4
    labels[4:, 0:2, 6:] = 5
    labels[4:, 0:2, 0:2] = 6
    labels[4:, 5:, 0:2] = 7
    labels[:2, 5:, 0:2] = 8
    mroi = subdomain_from_array(labels - 1, affine=None)
    return mroi


def test_subdomain():
    """Test basic construction of multiple_roi
    """
    mroi = make_subdomain()
    assert_equal(mroi.k, 8)


def test_subdomain2():
    """Test mroi.size
    """
    mroi = make_subdomain()
    assert_equal(len(mroi.get_size()), 8)
    for k in mroi.get_id():
        assert_equal(mroi.get_size(k),
                     np.sum(mroi.label == mroi.select_id(k)))


def test_copy_subdomain():
    """Test basic construction of multiple_roi
    """
    mroi = make_subdomain()
    foo_feature = [[i] * j for i, j in enumerate(mroi.get_size())]
    foo_roi_feature = np.arange(mroi.k)
    mroi.set_feature('a', foo_feature)
    mroi.set_roi_feature('b', foo_roi_feature)
    mroi_copy = mroi.copy()
    # check some properties of mroi
    assert_equal(mroi.k, 8)
    for k in mroi.get_id():
        assert_equal(mroi.get_feature('a', k), foo_feature[mroi.select_id(k)])
    assert_equal(mroi.get_roi_feature('b'), foo_roi_feature)
    # delete mroi
    del mroi
    # check mroi_copy
    assert_equal(mroi_copy.k, 8)
    for k in mroi_copy.get_id():
        assert_equal(mroi_copy.get_feature('a', k),
                     foo_feature[mroi_copy.select_id(k)])
    assert_equal(mroi_copy.get_roi_feature('b'), foo_roi_feature)


def test_select_roi():
    """
    """
    mroi = make_subdomain()
    aux = np.random.randn(np.prod(shape))
    data = [aux[mroi.label == k] for k in range(8)]
    mroi.set_feature('data', data)
    mroi.set_roi_feature('data_mean', range(8))
    mroi.select_roi([0])
    assert(mroi.k == 1)
    assert_equal(mroi.get_roi_feature('data_mean', 0), 0)

def test_roi_features():
    """
    """
    mroi = make_subdomain()
    aux = np.random.randn(np.prod(shape))
    dshape = (8, 3)
    data = np.random.randn(*dshape)
    mroi.set_roi_feature('data_mean', data)
    assert mroi.roi_features['data_mean'].shape == dshape

def test_subdomain_feature():
    """Test the basic construction of features
    """
    mroi = make_subdomain()
    aux = np.random.randn(np.prod(shape))
    data = [aux[mroi.label == k] for k in range(8)]
    mroi.set_feature('data', data)
    assert_equal(mroi.features['data'][0], data[0])


def test_sd_integrate():
    """Test the integration
    """
    mroi = make_subdomain()
    aux = np.random.randn(np.prod(shape))
    data = [aux[mroi.label == k] for k in range(8)]
    mroi.set_feature('data', data)
    sums = mroi.integrate('data')
    for k in range(8):
        assert_equal(sums[k], np.sum(data[k]))


def test_sd_integrate2():
    """Test the integration
    """
    mroi = make_subdomain()
    for k in mroi.get_id():
        assert_equal(mroi.get_volume(k), mroi.integrate(id=k))
    volume_from_integration = mroi.integrate()
    volume_from_feature = mroi.get_volume()
    for i in range(mroi.k):
        assert_equal(volume_from_feature[i], volume_from_integration[i])


def test_sd_representative():
    """Test the computation of representative features
    """
    mroi = make_subdomain()
    data = [[k] * mroi.get_size(k) for k in mroi.get_id()]
    mroi.set_feature('data', data)
    sums = mroi.representative_feature('data')
    for k in mroi.get_id():
        assert_equal(sums[mroi.select_id(k)], k)


def test_sd_from_ball():
    dom = domain_from_binary_array(np.ones((10, 10)))
    radii = np.array([2, 2, 2])
    positions = np.array([[3, 3], [3, 7], [7, 7]])
    subdomain = subdomain_from_balls(dom, positions, radii)
    assert_equal(subdomain.k, 3)
    assert_equal(subdomain.get_size(), np.array([9, 9, 9]))


def test_set_feature():
    """Test the feature building capability
    """
    mroi = make_subdomain()
    data = np.random.randn(np.prod(shape))
    feature_data = [data[mroi.select_id(k, roi=False)]
                    for k in mroi.get_id()]
    mroi.set_feature('data', feature_data)
    get_feature_output = mroi.get_feature('data')
    assert_equal([len(k) for k in mroi.get_feature('data')],
                 mroi.get_size())
    for k in mroi.get_id():
        assert_equal(mroi.get_feature('data', k),
                     data[mroi.select_id(k, roi=False)])
        assert_equal(get_feature_output[k],
                     data[mroi.select_id(k, roi=False)])


def test_set_feature2():
    """
    """
    mroi = make_subdomain()
    data = np.random.randn(np.prod(shape))
    feature_data = [data[mroi.select_id(k, roi=False)]
                    for k in mroi.get_id()]
    mroi.set_feature('data', feature_data)
    mroi.set_feature('data', np.asarray([1000]), id=0, override=True)
    assert_equal(mroi.get_feature('data', 0), [1000])


def test_get_coord():
    """
    """
    mroi = make_subdomain()
    for k in mroi.get_id():
        assert_equal(mroi.get_coord(k),
                     mroi.domain.coord[mroi.select_id(k, roi=False)])

if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
