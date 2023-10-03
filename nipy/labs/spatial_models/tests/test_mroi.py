# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the discrete_domain utilities.

Caveat assumes that the MNI template image is available at
in ~/.nipy/tests/data
"""

from os.path import dirname
from os.path import join as pjoin

import numpy as np
from nibabel import Nifti1Image, load
from numpy.testing import assert_almost_equal, assert_array_equal

from nipy.io.nibcompat import get_affine

from ..discrete_domain import domain_from_binary_array, grid_domain_from_image
from ..hroi import HROI_as_discrete_domain_blobs
from ..mroi import subdomain_from_array, subdomain_from_balls

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
    assert mroi.k == 8


def test_subdomain2():
    """Test mroi.size
    """
    mroi = make_subdomain()
    assert len(mroi.get_size()) == 8
    for k in mroi.get_id():
        assert (mroi.get_size(k) ==
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
    assert mroi.k == 8
    for k in mroi.get_id():
        assert_array_equal(mroi.get_feature('a', k), foo_feature[mroi.select_id(k)])
    assert_array_equal(mroi.get_roi_feature('b'), foo_roi_feature)
    # delete mroi
    del mroi
    # check mroi_copy
    assert mroi_copy.k == 8
    for k in mroi_copy.get_id():
        assert_array_equal(mroi_copy.get_feature('a', k),
                     foo_feature[mroi_copy.select_id(k)])
    assert_array_equal(mroi_copy.get_roi_feature('b'), foo_roi_feature)


def test_select_roi():
    # Test select_roi method
    mroi = make_subdomain()
    aux = np.random.randn(np.prod(shape))
    data = [aux[mroi.label == k] for k in range(8)]
    mroi.set_feature('data', data)
    mroi.set_roi_feature('data_mean', list(range(8)))
    mroi.select_roi([0])
    assert(mroi.k == 1)
    assert mroi.roi_features['id'] == [0]
    assert mroi.get_roi_feature('data_mean', 0) == 0
    mroi.select_roi([])
    assert(mroi.k == 0)
    assert list(mroi.roi_features) == ['id']
    assert list(mroi.roi_features['id']) == []


def test_roi_features():
    """
    """
    mroi = make_subdomain()
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
    assert mroi.features['data'][0] == data[0]


def test_sd_integrate():
    """Test the integration
    """
    mroi = make_subdomain()
    aux = np.random.randn(np.prod(shape))
    data = [aux[mroi.label == k] for k in range(8)]
    mroi.set_feature('data', data)
    sums = mroi.integrate('data')
    for k in range(8):
        assert sums[k] == np.sum(data[k])


def test_sd_integrate2():
    """Test the integration
    """
    mroi = make_subdomain()
    for k in mroi.get_id():
        assert mroi.get_volume(k) == mroi.integrate(id=k)
    volume_from_integration = mroi.integrate()
    volume_from_feature = mroi.get_volume()
    for i in range(mroi.k):
        assert volume_from_feature[i] == volume_from_integration[i]


def test_sd_representative():
    """Test the computation of representative features
    """
    mroi = make_subdomain()
    data = [[k] * mroi.get_size(k) for k in mroi.get_id()]
    mroi.set_feature('data', data)
    sums = mroi.representative_feature('data')
    for k in mroi.get_id():
        assert sums[mroi.select_id(k)] == k


def test_sd_from_ball():
    dom = domain_from_binary_array(np.ones((10, 10)))
    radii = np.array([2, 2, 2])
    positions = np.array([[3, 3], [3, 7], [7, 7]])
    subdomain = subdomain_from_balls(dom, positions, radii)
    assert subdomain.k == 3
    assert_array_equal(subdomain.get_size(), np.array([9, 9, 9]))


def test_set_feature():
    """Test the feature building capability
    """
    mroi = make_subdomain()
    data = np.random.randn(np.prod(shape))
    feature_data = [data[mroi.select_id(k, roi=False)]
                    for k in mroi.get_id()]
    mroi.set_feature('data', feature_data)
    get_feature_output = mroi.get_feature('data')
    assert_array_equal([len(k) for k in mroi.get_feature('data')],
                 mroi.get_size())
    for k in mroi.get_id():
        assert_array_equal(mroi.get_feature('data', k),
                     data[mroi.select_id(k, roi=False)])
        assert_array_equal(get_feature_output[k],
                     data[mroi.select_id(k, roi=False)])


def test_set_feature2():
    mroi = make_subdomain()
    data = np.random.randn(np.prod(shape))
    feature_data = [data[mroi.select_id(k, roi=False)]
                    for k in mroi.get_id()]
    mroi.set_feature('data', feature_data)
    mroi.set_feature('data', np.asarray([1000]), id=0, override=True)
    assert mroi.get_feature('data', 0) == [1000]


def test_get_coord():
    mroi = make_subdomain()
    for k in mroi.get_id():
        assert_array_equal(mroi.get_coord(k),
                     mroi.domain.coord[mroi.select_id(k, roi=False)])


def test_example():
    # Test example runs correctly
    eg_img = pjoin(dirname(__file__), 'some_blobs.nii')
    nim = load(eg_img)
    arr = nim.get_fdata() ** 2 > 0
    mask_image = Nifti1Image(arr.astype('u1'), get_affine(nim))
    domain = grid_domain_from_image(mask_image)
    data = nim.get_fdata()
    values = data[data != 0]

    # parameters
    threshold = 3.0 # blob-forming threshold
    smin = 5 # size threshold on blobs

    # compute the  nested roi object
    nroi = HROI_as_discrete_domain_blobs(domain, values, threshold=threshold,
                                         smin=smin)
    # compute region-level activation averages
    activation = [values[nroi.select_id(id, roi=False)]
                  for id in nroi.get_id()]
    nroi.set_feature('activation', activation)
    average_activation = nroi.representative_feature('activation')
    averages = [blob.mean() for blob in nroi.get_feature('activation')]
    assert_almost_equal(averages, average_activation, 6)
    # Test repeat
    assert_array_equal(average_activation, nroi.representative_feature('activation'))
    # Binary image is default
    bin_wim = nroi.to_image()
    bin_vox = bin_wim.get_fdata()
    assert_array_equal(np.unique(bin_vox), [0, 1])
    id_wim = nroi.to_image('id', roi=True, descrip='description')
    id_vox = id_wim.get_fdata()
    mask = bin_vox.astype(bool)
    assert_array_equal(id_vox[~mask], -1)
    ids = nroi.get_id()
    assert_array_equal(np.unique(id_vox), [-1] + list(ids))
    # Test activation
    wim = nroi.to_image('activation', roi=True, descrip='description')
    # Sadly, all cast to int
    assert_array_equal(np.unique(wim.get_fdata().astype(np.int32)), [-1, 3, 4, 5])
    # end blobs or leaves
    lroi = nroi.copy()
    lroi.reduce_to_leaves()
    assert lroi.k == 14
    assert len(lroi.get_feature('activation')) == lroi.k
