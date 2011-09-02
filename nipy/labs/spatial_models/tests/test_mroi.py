# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the discrete_domain utilities.

Caveat assumes that the MNI template image is available at
in ~/.nipy/tests/data
"""

import numpy as np
from ..mroi import *
from ..discrete_domain import domain_from_array

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
    """Test basic constructio of mulitple_roi
    """
    mroi = make_subdomain()
    assert mroi.k == 8


def test_subdomain2():
    """ Test mroi.size
    """
    subdomain = make_subdomain()
    assert len(subdomain.size) == 8
    for k in range(8):
        assert subdomain.size[k] == np.sum(subdomain.label == k)


def test_subdomain_feature():
    """Test the basic construction of features
    """
    subdomain = make_subdomain()
    aux = np.random.randn(np.prod(shape))
    data = [aux[subdomain.label == k] for k in range(8)]
    subdomain.set_feature('data', data)
    assert (subdomain.features['data'][0] == data[0]).all()


def test_sd_integrate():
    """Test the integration
    """
    subdomain = make_subdomain()
    aux = np.random.randn(np.prod(shape))
    data = [aux[subdomain.label == k] for k in range(8)]
    subdomain.set_feature('data', data)
    sums = subdomain.integrate('data')
    for k in range(8):
        assert sums[k] == np.sum(data[k])


def test_sd_representative():
    """Test the computation of representative features
    """
    subdomain = make_subdomain()
    aux = np.random.randn(np.prod(shape))
    data = [aux[subdomain.label == k] for k in range(8)]
    subdomain.set_feature('data', data)
    sums = subdomain.representative_feature('data')
    for k in range(8):
        assert sums[k] == np.mean(data[k])


def test_sd_from_ball():
    dom = domain_from_array(np.ones((10, 10)))
    radii = np.array([2, 2, 2])
    positions = np.array([[3, 3], [3, 7], [7, 7]])
    subdomain = subdomain_from_balls(dom, positions, radii)
    assert subdomain.k == 3
    assert (subdomain.size == np.array([9, 9, 9])).all()


def test_make_feature():
    """ tust the feature building capability
    """
    subdomain = make_subdomain()
    data = np.random.randn(np.prod(shape))
    subdomain.make_feature('data', data)
    print [i.shape for i in subdomain.get_feature('data')]
    assert (subdomain.features['data'][3] == data[subdomain.label == 3]).all()


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
