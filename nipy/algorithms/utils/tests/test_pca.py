# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np

from ..pca import pca
from nipy.io.api import  load_image
from nipy.testing import (assert_equal, assert_almost_equal,
                          assert_array_almost_equal, funcfile, assert_true,
                          assert_array_equal, assert_raises)


data = {}


def setup():
    img = load_image(funcfile)
    arr = img.get_data()
    #arr = np.rollaxis(arr, 3)
    data['nimages'] = arr.shape[3]
    data['fmridata'] = arr
    frame = data['fmridata'][...,0]
    data['mask'] = (frame > 500).astype(np.float64)


def reconstruct(time_series, images, axis=0):
    # Reconstruct data from remaining components
    n_tps = time_series.shape[0]
    images = np.rollaxis(images, axis)
    ncomps = images.shape[0]
    img_size = np.prod(images.shape[1:])
    rarr = images.reshape((ncomps, img_size))
    recond = np.dot(time_series, rarr)
    recond = recond.reshape((n_tps,) + images.shape[1:])
    if axis < 0:
        axis = axis + images.ndim
    recond = np.rollaxis(recond, 0, axis+1)
    return recond


def root_mse(arr, axis=0):
    return np.sqrt(np.square(arr).sum(axis=axis) / arr.shape[axis])


def pos1pca(arr, axis=0, **kwargs):
    ''' Return basis vectors and projections with first row positive '''
    res = pca(arr, axis, **kwargs)
    return res2pos1(res)


def res2pos1(res):
    # Orient basis vectors in standard direction
    axis = res['axis']
    bvs = res['basis_vectors']
    bps = res['basis_projections']
    signs = np.sign(bvs[0])
    res['basis_vectors'] = bvs * signs
    new_axes = [None] * bps.ndim
    n_comps = res['basis_projections'].shape[axis]
    new_axes[axis] = slice(0,n_comps)
    res['basis_projections'] = bps * signs[new_axes]
    return res


def test_same_basis():
    arr4d = data['fmridata']
    shp = arr4d.shape
    arr2d =  arr4d.reshape((np.prod(shp[:3]), shp[3]))
    res = pos1pca(arr2d, axis=-1)
    p1b_0 = res['basis_vectors']
    for i in range(3):
        res_again = pos1pca(arr2d, axis=-1)
        assert_almost_equal(res_again['basis_vectors'], p1b_0)


def test_2d_eq_4d():
    arr4d = data['fmridata']
    shp = arr4d.shape
    arr2d =  arr4d.reshape((np.prod(shp[:3]), shp[3]))
    arr3d = arr4d.reshape((shp[0], -1, shp[3]))
    res4d = pos1pca(arr4d, axis=-1, standardize=False)
    res3d = pos1pca(arr3d, axis=-1, standardize=False)
    res2d = pos1pca(arr2d, axis=-1, standardize=False)
    assert_array_almost_equal(res4d['basis_vectors'],
                              res2d['basis_vectors'])
    assert_array_almost_equal(res4d['basis_vectors'],
                              res3d['basis_vectors'])


def test_input_effects():
    # Test effects of axis specifications
    ntotal = data['nimages'] - 1
    # return full rank - mean PCA over last axis
    p = pos1pca(data['fmridata'], -1)
    assert_equal(p['basis_vectors'].shape, (data['nimages'], ntotal))
    assert_equal(p['basis_projections'].shape, data['mask'].shape + (ntotal,))
    assert_equal(p['pcnt_var'].shape, (ntotal,))
    # Reconstructed data lacks only mean
    rarr = reconstruct(p['basis_vectors'], p['basis_projections'], -1)
    rarr = rarr + data['fmridata'].mean(-1)[...,None]
    # same effect if over axis 0, which is the default
    arr = data['fmridata']
    arr = np.rollaxis(arr, -1)
    # Same basis once we've normalized the signs
    pr = pos1pca(arr)
    out_arr = np.rollaxis(pr['basis_projections'], 0, 4)
    assert_almost_equal(out_arr, p['basis_projections'])
    assert_almost_equal(p['basis_vectors'], pr['basis_vectors'])
    assert_almost_equal(p['pcnt_var'], pr['pcnt_var'])
    # Check axis None raises error
    assert_raises(ValueError, pca, data['fmridata'], None)


def test_diagonality():
    # basis_projections are diagonal, whether standarized or not
    p = pca(data['fmridata'], -1) # standardized
    assert_true(diagonal_covariance(p['basis_projections'], -1))
    pns = pca(data['fmridata'], -1, standardize=False) # not 
    assert_true(diagonal_covariance(pns['basis_projections'], -1))


def diagonal_covariance(arr, axis=0):
    arr = np.rollaxis(arr, axis)
    arr = arr.reshape(arr.shape[0], -1)
    aTa = np.dot(arr, arr.T)
    return np.allclose(aTa, np.diag(np.diag(aTa)), atol=1e-6)


def test_2D():
    # check that a standard 2D PCA works too
    M = 100
    N = 20
    L = M-1 # rank after mean removal
    data = np.random.uniform(size=(M, N))
    p = pca(data)
    ts = p['basis_vectors']
    imgs = p['basis_projections']
    assert_equal(ts.shape, (M, L))
    assert_equal(imgs.shape, (L, N))
    rimgs = reconstruct(ts, imgs)
    # add back the sqrt MSE, because we standardized
    data_mean = data.mean(0)[None,...]
    demeaned = data - data_mean
    rmse = root_mse(demeaned, axis=0)[None,...]
    # also add back the mean
    assert_array_almost_equal((rimgs * rmse) + data_mean, data)
    # if standardize is set, or not, covariance is diagonal
    assert_true(diagonal_covariance(imgs))
    p = pca(data, standardize=False)
    imgs = p['basis_projections']
    assert_true(diagonal_covariance(imgs))


def test_PCAMask():
    # for 2 and 4D case
    ntotal = data['nimages'] - 1
    ncomp = 5
    arr4d = data['fmridata']
    mask3d = data['mask']
    arr2d = arr4d.reshape((-1, data['nimages']))
    mask1d = mask3d.reshape((-1))
    for arr, mask in (arr4d, mask3d), (arr2d, mask1d):
        p = pca(arr, -1, mask, ncomp=ncomp)
        assert_equal(p['basis_vectors'].shape, (data['nimages'], ntotal))
        assert_equal(p['basis_projections'].shape, mask.shape + (ncomp,))
        assert_equal(p['pcnt_var'].shape, (ntotal,))
        assert_almost_equal(p['pcnt_var'].sum(), 100.) 


def test_PCAMask_nostandardize():
    ntotal = data['nimages'] - 1
    ncomp = 5
    p = pca(data['fmridata'], -1, data['mask'], ncomp=ncomp,
            standardize=False)
    assert_equal(p['basis_vectors'].shape, (data['nimages'], ntotal))
    assert_equal(p['basis_projections'].shape, data['mask'].shape + (ncomp,))
    assert_equal(p['pcnt_var'].shape, (ntotal,))
    assert_almost_equal(p['pcnt_var'].sum(), 100.)


def test_PCANoMask():
    ntotal = data['nimages'] - 1
    ncomp = 5
    p = pca(data['fmridata'], -1, ncomp=ncomp)
    assert_equal(p['basis_vectors'].shape, (data['nimages'], ntotal))
    assert_equal(p['basis_projections'].shape,  data['mask'].shape + (ncomp,))
    assert_equal(p['pcnt_var'].shape, (ntotal,))
    assert_almost_equal(p['pcnt_var'].sum(), 100.)


def test_PCANoMask_nostandardize():
    ntotal = data['nimages'] - 1
    ncomp = 5
    p = pca(data['fmridata'], -1, ncomp=ncomp, standardize=False)
    assert_equal(p['basis_vectors'].shape, (data['nimages'], ntotal))
    assert_equal(p['basis_projections'].shape,  data['mask'].shape + (ncomp,))
    assert_equal(p['pcnt_var'].shape, (ntotal,))
    assert_almost_equal(p['pcnt_var'].sum(), 100.)


def test_keep():
    # Data is projected onto k=10 dimensional subspace
    # then has its mean removed.
    # Should still have rank 10.
    k = 10
    ncomp = 5
    ntotal = k
    X = np.random.standard_normal((data['nimages'], k))
    p = pca(data['fmridata'], -1, ncomp=ncomp, design_keep=X)
    assert_equal(p['basis_vectors'].shape, (data['nimages'], ntotal))
    assert_equal(p['basis_projections'].shape,  data['mask'].shape + (ncomp,))
    assert_equal(p['pcnt_var'].shape, (ntotal,))
    assert_almost_equal(p['pcnt_var'].sum(), 100.)


def test_resid():
    # Data is projected onto k=10 dimensional subspace then has its mean
    # removed.  Should still have rank 10.
    k = 10
    ncomp = 5
    ntotal = k
    X = np.random.standard_normal((data['nimages'], k))
    p = pca(data['fmridata'], -1, ncomp=ncomp, design_resid=X)
    assert_equal(p['basis_vectors'].shape, (data['nimages'], ntotal))
    assert_equal(p['basis_projections'].shape, data['mask'].shape + (ncomp,))
    assert_equal(p['pcnt_var'].shape, (ntotal,))
    assert_almost_equal(p['pcnt_var'].sum(), 100.)
    # if design_resid is None, we do not remove the mean, and we get
    # full rank from our data
    p = pca(data['fmridata'], -1, design_resid=None)
    rank = p['basis_vectors'].shape[1]
    assert_equal(rank, data['nimages'])
    rarr = reconstruct(p['basis_vectors'], p['basis_projections'], -1)
    # add back the sqrt MSE, because we standardized
    rmse = root_mse(data['fmridata'], axis=-1)[...,None]
    assert_true(np.allclose(rarr * rmse, data['fmridata']))


def test_both():
    k1 = 10
    k2 = 8
    ncomp = 5
    ntotal = k1
    X1 = np.random.standard_normal((data['nimages'], k1))
    X2 = np.random.standard_normal((data['nimages'], k2))
    p = pca(data['fmridata'], -1, ncomp=ncomp, design_resid=X2, design_keep=X1)
    assert_equal(p['basis_vectors'].shape, (data['nimages'], ntotal))
    assert_equal(p['basis_projections'].shape,  data['mask'].shape + (ncomp,))
    assert_equal(p['pcnt_var'].shape, (ntotal,))
    assert_almost_equal(p['pcnt_var'].sum(), 100.)
