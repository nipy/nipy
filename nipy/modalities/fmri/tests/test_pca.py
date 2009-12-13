import numpy as np

from nipy.modalities.fmri.pca import pca
from nipy.io.api import  load_image
from nipy.testing import assert_equal, assert_almost_equal, \
    assert_array_almost_equal, funcfile, parametric, \
    assert_array_equal, assert_raises


data = {}


def setup():
    img = load_image(funcfile)
    arr = np.array(img)
    #arr = np.rollaxis(arr, 3)
    data['nimages'] = arr.shape[3]
    data['fmridata'] = arr
    frame = data['fmridata'][...,0]
    data['mask'] = (frame > 500).astype(np.float64)


@parametric
def test_input_effects():
    ntotal = data['nimages'] - 1
    # return full rank PCA over last axis by default
    p = pca(data['fmridata'])
    yield assert_equal, p['rank'], ntotal
    yield assert_equal, p['time_series'].shape, (ntotal, data['nimages'])
    yield assert_equal, p['images'].shape, data['mask'].shape + (ntotal,)
    yield assert_equal, p['pcnt_var'].shape, (ntotal,)
    # same effect if over axis 0
    arr = data['fmridata']
    arr = np.rollaxis(arr, -1)
    pr = pca(data['fmridata'], 0)
    out_arr = np.rollaxis(pr['images'], 0, 3)
    yield assert_array_equal, out_arr, p['images']
    yield assert_array_equal, p['time_series'], pr['time_series']
    yield assert_array_equal, p['pcnt_var'], pr['pcnt_var']
    # Check axis None raises error
    yield assert_raises, ValueError, pca, data['fmridata'], None


def test_PCAMask():
    ntotal = data['nimages'] - 1
    ncomp = 5
    p = pca(data['fmridata'], -1, data['mask'], ncomp=ncomp)
    yield assert_equal, p['rank'], ntotal
    yield assert_equal, p['time_series'].shape, (ncomp, data['nimages'])
    yield assert_equal, p['images'].shape, data['mask'].shape + (ncomp,)
    yield assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield assert_almost_equal, p['pcnt_var'].sum(), 100.
    

def test_PCAMask_nostandardize():
    ntotal = data['nimages'] - 1
    ncomp = 5
    p = pca(data['fmridata'], -1, data['mask'], ncomp=ncomp,
            standardize=False)
    yield assert_equal, p['rank'], ntotal
    yield assert_equal, p['time_series'].shape, (ncomp, data['nimages'])
    yield assert_equal, p['images'].shape, data['mask'].shape + (ncomp,)
    yield assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield assert_almost_equal, p['pcnt_var'].sum(), 100.


def test_PCANoMask():
    ntotal = data['nimages'] - 1
    ncomp = 5
    p = pca(data['fmridata'], ncomp=ncomp)
    yield assert_equal, p['rank'], ntotal
    yield assert_equal, p['time_series'].shape, (ncomp, data['nimages'])
    yield assert_equal, p['images'].shape,  data['mask'].shape + (ncomp,)
    yield assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield assert_almost_equal, p['pcnt_var'].sum(), 100.


def test_PCANoMask_nostandardize():
    ntotal = data['nimages'] - 1
    ncomp = 5
    p = pca(data['fmridata'], ncomp=ncomp, standardize=False)
    yield assert_equal, p['rank'], ntotal
    yield assert_equal, p['time_series'].shape, (ncomp, data['nimages'])
    yield assert_equal, p['images'].shape,  data['mask'].shape + (ncomp,)
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
    p = pca(data['fmridata'], ncomp=ncomp, design_keep=X)
    yield assert_equal, p['rank'], ntotal
    yield assert_equal, p['time_series'].shape, (ncomp, data['nimages'])
    yield assert_equal, p['images'].shape,  data['mask'].shape + (ncomp,)
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
    p = pca(data['fmridata'], ncomp=ncomp, design_resid=X)
    yield assert_equal, p['rank'], ntotal
    yield assert_equal, p['time_series'].shape, (ncomp, data['nimages'])
    yield assert_equal, p['images'].shape, data['mask'].shape + (ncomp,)
    yield assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield assert_almost_equal, p['pcnt_var'].sum(), 100.
    # if design_resid is None, we do not remove the mean
    p = pca(data['fmridata'], ncomp=ncomp, design_resid=None)
    yield assert_equal, p['rank'], data['nimages']


def test_both():
    k1 = 10
    k2 = 8
    ncomp = 5
    ntotal = k1
    X1 = np.random.standard_normal((data['nimages'], k1))
    X2 = np.random.standard_normal((data['nimages'], k2))
    p = pca(data['fmridata'], ncomp=ncomp, design_resid=X2, design_keep=X1)
    yield assert_equal, p['rank'], ntotal
    yield assert_equal, p['time_series'].shape, (ncomp, data['nimages'])
    yield assert_equal, p['images'].shape,  data['mask'].shape + (ncomp,)
    yield assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield assert_almost_equal, p['pcnt_var'].sum(), 100.
