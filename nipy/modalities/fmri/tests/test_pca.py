import numpy as np

from nipy.modalities.fmri.api import FmriImageList
from nipy.modalities.fmri.pca import pca
from nipy.io.api import  load_image
import nipy.testing as niptest

data = {}

def setup():
    img = load_image(niptest.funcfile)
    data['fmridata'] = FmriImageList.from_image(img)
    frame = data['fmridata'][0]
    data['mask'] = img.get_lookalike(
                    np.greater(frame.get_data(), 500).astype(np.float64))
    print data['mask'].get_data().shape, np.sum(data['mask'].get_data())

def test_PCAMask():
    nimages = len(data['fmridata'].list) 
    ntotal = nimages - 1
    ncomp = 5
    p = pca(data['fmridata'].get_data(), data['mask'].get_data(), ncomp=ncomp)

    yield niptest.assert_equal, p['rank'], ntotal
    yield niptest.assert_equal, p['time_series'].shape, (ncomp,
                                                         nimages)
    yield niptest.assert_equal, p['images'].get_data().shape, (ncomp,) + data['mask'].shape
    yield niptest.assert_equal, p['pcnt_var'].get_data().shape, (ntotal,)
    yield niptest.assert_almost_equal, p['pcnt_var'].sum(), 100.

def test_PCAMask_nostandardize():
    nimages = len(data['fmridata'].list) 
    ntotal = nimages - 1
    ncomp = 5
    p = pca(data['fmridata'], data['mask'], ncomp=ncomp, standardize=False)

    yield niptest.assert_equal, p['rank'], ntotal
    yield niptest.assert_equal, p['time_series'].get_data().shape, (ncomp,
                                                         nimages)
    yield niptest.assert_equal, p['images'].shape, (ncomp,) + data['mask'].shape
    yield niptest.assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield niptest.assert_almost_equal, p['pcnt_var'].sum(), 100.


def test_PCANoMask():
    nimages = len(data['fmridata'].list) 
    ntotal = nimages - 1
    ncomp = 5
    p = pca(data['fmridata'], ncomp=ncomp)

    yield niptest.assert_equal, p['rank'], ntotal
    yield niptest.assert_equal, p['time_series'].shape, (ncomp,
                                                         nimages)
    yield niptest.assert_equal, p['images'].shape, (ncomp,) + data['mask'].shape
    yield niptest.assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield niptest.assert_almost_equal, p['pcnt_var'].sum(), 100.

def test_PCANoMask_nostandardize():
    nimages = len(data['fmridata'].list) 
    ntotal = nimages - 1
    ncomp = 5
    p = pca(data['fmridata'], ncomp=ncomp, standardize=False)

    yield niptest.assert_equal, p['rank'], ntotal
    yield niptest.assert_equal, p['time_series'].shape, (ncomp,
                                                         nimages)
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
    nimages = len(data['fmridata'].list) 
    ntotal = k
    X = np.random.standard_normal((nimages, k))
    p = pca(data['fmridata'], ncomp=ncomp, design_keep=X)

    yield niptest.assert_equal, p['rank'], ntotal
    yield niptest.assert_equal, p['time_series'].shape, (ncomp,
                                                         nimages)
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
    nimages = len(data['fmridata'].list) 
    ntotal = k
    X = np.random.standard_normal((nimages, k))
    p = pca(data['fmridata'], ncomp=ncomp, design_resid=X)

    yield niptest.assert_equal, p['rank'], ntotal
    yield niptest.assert_equal, p['time_series'].shape, (ncomp,
                                                         nimages)
    yield niptest.assert_equal, p['images'].shape, (ncomp,) + data['mask'].shape
    yield niptest.assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield niptest.assert_almost_equal, p['pcnt_var'].sum(), 100.





def test_both():
    k1 = 10
    k2 = 8
    ncomp = 5
    nimages = len(data['fmridata'].list) 
    ntotal = k1
    X1 = np.random.standard_normal((nimages, k1))
    X2 = np.random.standard_normal((nimages, k2))
    p = pca(data['fmridata'], ncomp=ncomp, design_resid=X2, design_keep=X1)

    yield niptest.assert_equal, p['rank'], ntotal
    yield niptest.assert_equal, p['time_series'].shape, (ncomp,
                                                         nimages)
    yield niptest.assert_equal, p['images'].shape, (ncomp,) + data['mask'].shape
    yield niptest.assert_equal, p['pcnt_var'].shape, (ntotal,)
    yield niptest.assert_almost_equal, p['pcnt_var'].sum(), 100.





        


