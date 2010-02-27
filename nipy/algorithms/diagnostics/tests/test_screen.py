""" Testing diagnostic screen

"""

import numpy as np

import nipy as ni
import nipy.algorithms.diagnostics as nad

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nipy.testing import parametric, funcfile, anatfile


@parametric
def test_screen():
    img = ni.load_image(funcfile)
    res = nad.screen(img)
    yield assert_equal(res['mean'].ndim, 3)
    yield assert_equal(res['pca'].ndim, 4)
    yield assert_equal(sorted(res.keys()),
                       ['max', 'mean', 'min',
                        'pca', 'pca_res',
                        'std', 'ts_res'])
    data = np.asarray(img)
    yield assert_array_equal(np.max(data, axis=-1),
                             res['max'])
    yield assert_array_equal(np.mean(data, axis=-1),
                             res['mean'])
    yield assert_array_equal(np.min(data, axis=-1),
                             res['min'])
    yield assert_array_equal(np.std(data, axis=-1),
                             res['std'])
    pca_res = nad.pca.pca(data, axis=-1, standardize=False, ncomp=10)
    for key in pca_res:
        yield assert_array_equal(pca_res[key], res['pca_res'][key])
    ts_res = nad.time_slice_diffs(data)
    for key in ts_res:
        yield assert_array_equal(ts_res[key], res['ts_res'][key])
    
