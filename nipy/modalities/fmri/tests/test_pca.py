import numpy as np

from nipy.modalities.fmri.api import FmriImageList
from nipy.modalities.fmri.pca import pca
from nipy.core.api import Image
from nipy.io.api import  load_image
import nipy.testing as niptest

data = {}

def setup():
    img = load_image(niptest.funcfile)
    arr = np.array(img)
    arr = np.rollaxis(arr, 3)
    data['nimages'] = arr.shape[0]
    data['fmridata'] = arr
    frame = data['fmridata'][0]
    data['mask'] = (frame > 500).astype(np.float64)


def test_PCAMask():
    ntotal = data['nimages'] - 1
    ncomp = 5
    p = pca(data['fmridata'], data['mask'], ncomp=ncomp)
    yield niptest.assert_equal, p['rank'], ntotal
    yield niptest.assert_equal, p['time_series'].shape, (ncomp,
                                                         data['nimages'])
    yield niptest.assert_equal, p['images'].shape, (ncomp,) + data['mask'].shape
    yield niptest.assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield niptest.assert_almost_equal, p['pcnt_var'].sum(), 100.

def test_PCAMask_nostandardize():
    ntotal = data['nimages'] - 1
    ncomp = 5
    p = pca(data['fmridata'], data['mask'], ncomp=ncomp, standardize=False)

    yield niptest.assert_equal, p['rank'], ntotal
    yield niptest.assert_equal, p['time_series'].shape, (ncomp,
                                                         data['nimages'])
    yield niptest.assert_equal, p['images'].shape, (ncomp,) + data['mask'].shape
    yield niptest.assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield niptest.assert_almost_equal, p['pcnt_var'].sum(), 100.


def test_PCANoMask():
    ntotal = data['nimages'] - 1
    ncomp = 5
    p = pca(data['fmridata'], ncomp=ncomp)

    yield niptest.assert_equal, p['rank'], ntotal
    yield niptest.assert_equal, p['time_series'].shape, (ncomp,
                                                         data['nimages'])
    yield niptest.assert_equal, p['images'].shape, (ncomp,) + data['mask'].shape
    yield niptest.assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield niptest.assert_almost_equal, p['pcnt_var'].sum(), 100.

def test_PCANoMask_nostandardize():
    ntotal = data['nimages'] - 1
    ncomp = 5
    p = pca(data['fmridata'], ncomp=ncomp, standardize=False)

    yield niptest.assert_equal, p['rank'], ntotal
    yield niptest.assert_equal, p['time_series'].shape, (ncomp,
                                                         data['nimages'])
    yield niptest.assert_equal, p['images'].shape, (ncomp,) + data['mask'].shape
    yield niptest.assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield niptest.assert_almost_equal, p['pcnt_var'].sum(), 100.

def test_keep():
    """
    Data is projected onto k=10 dimensional subspace
    then has its mean removed.
    Should still have rank 10.

    """
    k = 10
    ncomp = 5
    ntotal = k
    X = np.random.standard_normal((data['nimages'], k))
    p = pca(data['fmridata'], ncomp=ncomp, design_keep=X)

    yield niptest.assert_equal, p['rank'], ntotal
    yield niptest.assert_equal, p['time_series'].shape, (ncomp,
                                                         data['nimages'])
    yield niptest.assert_equal, p['images'].shape, (ncomp,) + data['mask'].shape
    yield niptest.assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield niptest.assert_almost_equal, p['pcnt_var'].sum(), 100.

def test_resid():
    """
    Data is projected onto k=10 dimensional subspace
    then has its mean removed.
    Should still have rank 10.

    """
    k = 10
    ncomp = 5
    ntotal = k
    X = np.random.standard_normal((data['nimages'], k))
    p = pca(data['fmridata'], ncomp=ncomp, design_resid=X)

    yield niptest.assert_equal, p['rank'], ntotal
    yield niptest.assert_equal, p['time_series'].shape, (ncomp,
                                                         data['nimages'])
    yield niptest.assert_equal, p['images'].shape, (ncomp,) + data['mask'].shape
    yield niptest.assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield niptest.assert_almost_equal, p['pcnt_var'].sum(), 100.





def test_both():
    k1 = 10
    k2 = 8
    ncomp = 5
    ntotal = k1
    X1 = np.random.standard_normal((data['nimages'], k1))
    X2 = np.random.standard_normal((data['nimages'], k2))
    p = pca(data['fmridata'], ncomp=ncomp, design_resid=X2, design_keep=X1)

    yield niptest.assert_equal, p['rank'], ntotal
    yield niptest.assert_equal, p['time_series'].shape, (ncomp,
                                                         data['nimages'])
    yield niptest.assert_equal, p['images'].shape, (ncomp,) + data['mask'].shape
    yield niptest.assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield niptest.assert_almost_equal, p['pcnt_var'].sum(), 100.





        


