# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the discrete_domain utilities.

Caveat assumes that the MNI template image is available at
in ~/.nipy/tests/data
"""

import numpy as np
from numpy.testing import assert_almost_equal
from nipy.neurospin.spatial_models.discrete_domain import *
from nibabel import Nifti1Image

shape = np.array([5, 6, 7, 8, 9])

def generate_dataset(shape):
    """Generate a dataset with the repscribed dhape
    """
    dim = len(shape)
    idx = np.reshape(np.indices(shape), (dim, -1)).T
    return idx

def test_smatrix_1d():
    """Test the 1-d topological domain
    """
    idx = generate_dataset(shape[:1])
    sm = smatrix_from_nd_idx(idx, nn=0)
    assert(sm.data.size==2*shape[0]-2)

def test_smatrix_2d():
    """Test the 2-d topological domain
    """
    idx = generate_dataset(shape[:2])
    sm = smatrix_from_nd_idx(idx, nn=0)
    ne = 2*(2*np.prod(shape[:2])-shape[0]-shape[1])
    assert(sm.data.size==ne)

def test_smatrix_3d():
    """Test the 3-d topological domain
    """
    idx = generate_dataset(shape[:3])
    sm = smatrix_from_nd_idx(idx)
    ne = 2*(3*np.prod(shape[:3])-shape[0]*shape[1]-shape[0]*shape[2]
            -shape[1]*shape[2])
    assert(sm.data.size==ne)

def test_smatrix_4d():
    """Test the 4-d topological domain
    """
    idx = generate_dataset(shape[:4])
    sm = smatrix_from_nd_idx(idx)
    ne = 4*np.prod(shape[:4])
    for d in range(4):
        ne -= np.prod(shape[:4])/shape[d]
    ne *= 2
    assert(sm.data.size==ne)

def test_smatrix_5d():
    """Test the 5-d topological domain
    """
    idx = generate_dataset(shape)
    sm = smatrix_from_nd_idx(idx)
    ne = 5*np.prod(shape)
    for d in range(5):
        ne -= np.prod(shape)/shape[d]
    ne *= 2
    assert(sm.data.size==ne)

def test_smatrix_5d_bis():
    """Test the 5-d topological domain
    """
    toto = np.ones(shape)
    sm = smatrix_from_nd_array(toto)
    ne = 5*np.prod(shape)
    for d in range(5):
        ne -= np.prod(shape)/shape[d]
    ne *= 2
    assert(sm.data.size==ne)

def test_matrix_from_3d_array():
    """Test thetopology using the nipy.graph approach 
    """
    toto = np.ones(shape[:3])
    sm = smatrix_from_3d_array(toto, 6)
    ne = 3*np.prod(shape[:3])
    for d in range(3):
        ne -= np.prod(shape[:3])/shape[d]
    ne *= 2
    print sm.data, ne
    assert((sm.data>0).sum()==ne)

def test_array_domain():
    """Test the construction of domain based on array
    """
    toto = np.ones(shape)
    ddom = domain_from_array(toto)
    assert (np.sum(ddom.local_volume)==np.prod(shape))

def test_image_domain():
    """Test the construction of domain based on image
    """
    toto = np.ones(shape[:3])
    affine = np.random.randn(4,4)
    affine[3:,:3]=0
    nim = Nifti1Image(toto, affine)
    ddom = domain_from_image(nim)
    ref = np.sum(toto)*np.absolute(np.linalg.det(affine))
    assert_almost_equal (np.sum(ddom.local_volume), ref)
    
def test_array_grid_domain():
    """Test the construction of grid domain based on array
    """
    toto = np.ones(shape)
    ddom = grid_domain_from_array(toto)
    assert (np.sum(ddom.local_volume)==np.prod(shape))

def test_image_grid_domain():
    """Test the construction of grid domain based on image
    """
    toto = np.ones(shape[:3])
    affine = np.random.randn(4,4)
    affine[3:,:3]=0
    nim = Nifti1Image(toto, affine)
    ddom = grid_domain_from_image(nim)
    ref = np.sum(toto)*np.absolute(np.linalg.det(affine))
    assert_almost_equal (np.sum(ddom.local_volume), ref)

def test_feature():
    """ test feature inclusion
    """
    toto = np.random.rand(*shape)
    ddom = domain_from_array(toto)
    ddom.set_feature('data', np.ravel(toto))
    plop = ddom.get_feature('data')
    assert_almost_equal (plop, np.ravel(toto))

def test_mask_feature():
    """ test_feature_masking
    """
    toto = np.random.rand(*shape)
    ddom = domain_from_array(toto)
    ddom.set_feature('data', np.ravel(toto))
    mdom = ddom.mask(np.ravel(toto>.5))
    plop = mdom.get_feature('data')
    assert_almost_equal (plop, toto[toto>.5])
    
def test_domain_mask():
    """test domain masking
    """
    toto = np.random.rand(*shape)
    ddom = domain_from_array(toto)
    mdom = ddom.mask(np.ravel(toto>.5))
    assert (mdom.size==np.sum(toto>.5))
    
def test_grid_domain_mask():
    """test grid domain masking
    """
    toto = np.random.rand(*shape)
    ddom = grid_domain_from_array(toto)
    mdom = ddom.mask(np.ravel(toto>.5))
    assert (mdom.size==np.sum(toto>.5))

def test_representative():
    """ test representative computation
    """
    toto = np.random.rand(*shape)
    ddom = domain_from_array(toto)
    ddom.set_feature('data', np.ravel(toto))
    dmean = toto.mean()
    dmin = toto.min()
    dmax = toto.max()
    dmed = np.median(toto)
    assert_almost_equal(ddom.representative_feature('data', 'mean'), dmean)
    assert_almost_equal(ddom.representative_feature('data', 'min'), dmin)
    assert_almost_equal(ddom.representative_feature('data', 'max'), dmax)
    assert_almost_equal(ddom.representative_feature('data', 'median'), dmed)

def test_integrate_1d():
    """ test integration in 1d
    """
    toto = np.random.rand(*shape)
    ddom = domain_from_array(toto)
    ddom.set_feature('data', np.ravel(toto))
    assert_almost_equal(ddom.integrate('data'), toto.sum())
    

def test_integrate_2d():
    """test integration in 2d
    """
    toto = np.random.rand(*shape)
    ddom = domain_from_array(toto)
    ftoto = np.ravel(toto)
    f2 = np.vstack((ftoto, ftoto)).T
    ddom.set_feature('data', f2)
    ts = np.ones(2)*toto.sum()
    assert_almost_equal(ddom.integrate('data'), ts)
    
if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])





