# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np

from nipy.modalities.fmri.pca import pca
from nipy.io.api import  load_image
from nipy.testing import assert_equal, assert_almost_equal, \
    assert_array_almost_equal, funcfile, assert_true, \
    assert_array_equal, assert_raises, \
    parametric


data = {}


def setup():
    img = load_image(funcfile)
    arr = np.array(img)
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


@parametric
def test_input_effects():
    ntotal = data['nimages'] - 1
    # return full rank - mean PCA over last axis
    p = pca(data['fmridata'], -1)
    yield assert_equal(
        p['basis_vectors'].shape,
        (data['nimages'], ntotal))
    yield assert_equal(
        p['basis_projections'].shape,
        data['mask'].shape + (ntotal,))
    yield assert_equal(p['pcnt_var'].shape, (ntotal,))
    # Reconstructed data lacks only mean
    rarr = reconstruct(p['basis_vectors'], p['basis_projections'], -1)
    rarr = rarr + data['fmridata'].mean(-1)[...,None]
    # same effect if over axis 0, which is the default
    arr = data['fmridata']
    arr = np.rollaxis(arr, -1)
    pr = pca(arr)
    out_arr = np.rollaxis(pr['basis_projections'], 0, 4)
    yield assert_array_equal(out_arr, p['basis_projections'])
    yield assert_array_equal(p['basis_vectors'], pr['basis_vectors'])
    yield assert_array_equal(p['pcnt_var'], pr['pcnt_var'])
    # Check axis None raises error
    yield assert_raises(ValueError, pca, data['fmridata'], None)


@parametric
def test_diagonality():
    # basis_projections are diagonal, whether standarized or not
    p = pca(data['fmridata'], -1) # standardized
    yield assert_true(diagonal_covariance(p['basis_projections'], -1))
    pns = pca(data['fmridata'], -1, standardize=False) # not 
    yield assert_true(diagonal_covariance(pns['basis_projections'], -1))


def diagonal_covariance(arr, axis=0):
    arr = np.rollaxis(arr, axis)
    arr = arr.reshape(arr.shape[0], -1)
    aTa = np.dot(arr, arr.T)
    return np.allclose(aTa, np.diag(np.diag(aTa)), atol=1e-6)


@parametric
def test_2D():
    # check that a standard 2D PCA works too
    M = 100
    N = 20
    L = M-1 # rank after mean removal
    data = np.random.uniform(size=(M, N))
    p = pca(data)
    ts = p['basis_vectors']
    imgs = p['basis_projections']
    yield assert_equal(ts.shape, (M, L))
    yield assert_equal(imgs.shape, (L, N))
    rimgs = reconstruct(ts, imgs)
    # add back the sqrt MSE, because we standardized
    data_mean = data.mean(0)[None,...]
    demeaned = data - data_mean
    rmse = root_mse(demeaned, axis=0)[None,...]
    # also add back the mean
    yield assert_array_almost_equal((rimgs * rmse) + data_mean, data)
    # if standardize is set, or not, covariance is diagonal
    yield assert_true(diagonal_covariance(imgs))
    p = pca(data, standardize=False)
    imgs = p['basis_projections']
    yield assert_true(diagonal_covariance(imgs))
    

@parametric
def test_PCAMask():
    ntotal = data['nimages'] - 1
    ncomp = 5
    p = pca(data['fmridata'], -1, data['mask'], ncomp=ncomp)
    yield assert_equal(
        p['basis_vectors'].shape,
        (data['nimages'], ntotal))
    yield assert_equal(
        p['basis_projections'].shape,
        data['mask'].shape + (ncomp,))
    yield assert_equal(p['pcnt_var'].shape, (ntotal,))
    yield assert_almost_equal(p['pcnt_var'].sum(), 100.)


def test_PCAMask_nostandardize():
    ntotal = data['nimages'] - 1
    ncomp = 5
    p = pca(data['fmridata'], -1, data['mask'], ncomp=ncomp,
            standardize=False)
    yield assert_equal, p['basis_vectors'].shape, (data['nimages'], ntotal)
    yield assert_equal, p['basis_projections'].shape, data['mask'].shape + (ncomp,)
    yield assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield assert_almost_equal, p['pcnt_var'].sum(), 100.


def test_PCANoMask():
    ntotal = data['nimages'] - 1
    ncomp = 5
    p = pca(data['fmridata'], -1, ncomp=ncomp)
    yield assert_equal, p['basis_vectors'].shape, (data['nimages'], ntotal)
    yield assert_equal, p['basis_projections'].shape,  data['mask'].shape + (ncomp,)
    yield assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield assert_almost_equal, p['pcnt_var'].sum(), 100.


def test_PCANoMask_nostandardize():
    ntotal = data['nimages'] - 1
    ncomp = 5
    p = pca(data['fmridata'], -1, ncomp=ncomp, standardize=False)
    yield assert_equal, p['basis_vectors'].shape, (data['nimages'], ntotal)
    yield assert_equal, p['basis_projections'].shape,  data['mask'].shape + (ncomp,)
    yield assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield assert_almost_equal, p['pcnt_var'].sum(), 100.


def test_keep():
    # Data is projected onto k=10 dimensional subspace
    # then has its mean removed.
    # Should still have rank 10.
    k = 10
    ncomp = 5
    ntotal = k
    X = np.random.standard_normal((data['nimages'], k))
    p = pca(data['fmridata'], -1, ncomp=ncomp, design_keep=X)
    yield assert_equal, p['basis_vectors'].shape, (data['nimages'], ntotal)
    yield assert_equal, p['basis_projections'].shape,  data['mask'].shape + (ncomp,)
    yield assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield assert_almost_equal, p['pcnt_var'].sum(), 100.


@parametric
def test_resid():
    # Data is projected onto k=10 dimensional subspace then has its mean
    # removed.  Should still have rank 10.
    k = 10
    ncomp = 5
    ntotal = k
    X = np.random.standard_normal((data['nimages'], k))
    p = pca(data['fmridata'], -1, ncomp=ncomp, design_resid=X)
    yield assert_equal(
        p['basis_vectors'].shape,
        (data['nimages'], ntotal))
    yield assert_equal(
        p['basis_projections'].shape,
        data['mask'].shape + (ncomp,))
    yield assert_equal(p['pcnt_var'].shape, (ntotal,))
    yield assert_almost_equal(p['pcnt_var'].sum(), 100.)
    # if design_resid is None, we do not remove the mean, and we get
    # full rank from our data
    p = pca(data['fmridata'], -1, design_resid=None)
    rank = p['basis_vectors'].shape[1]
    yield assert_equal(rank, data['nimages'])
    rarr = reconstruct(p['basis_vectors'], p['basis_projections'], -1)
    # add back the sqrt MSE, because we standardized
    rmse = root_mse(data['fmridata'], axis=-1)[...,None]
    yield assert_array_almost_equal(rarr * rmse, data['fmridata'])


def test_both():
    k1 = 10
    k2 = 8
    ncomp = 5
    ntotal = k1
    X1 = np.random.standard_normal((data['nimages'], k1))
    X2 = np.random.standard_normal((data['nimages'], k2))
    p = pca(data['fmridata'], -1, ncomp=ncomp, design_resid=X2, design_keep=X1)
    yield assert_equal, p['basis_vectors'].shape, (data['nimages'], ntotal)
    yield assert_equal, p['basis_projections'].shape,  data['mask'].shape + (ncomp,)
    yield assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield assert_almost_equal, p['pcnt_var'].sum(), 100.
