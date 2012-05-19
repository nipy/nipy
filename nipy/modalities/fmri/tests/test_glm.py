# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the glm utilities.
"""

import numpy as np
from nose.tools import assert_true
from numpy.testing import assert_almost_equal
from ..glm import GLM, GLMResults, Contrast

def test_glm_ols():
    n, p, q = 100, 80, 10
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    result = GLM(X, 'ols').fit(Y)
    assert_true((result.labels == np.zeros(n)).all())
    assert_true(result.results.keys() == ['0'])
    assert_true(result.results['0'].theta.shape == (q, n))
    assert_almost_equal(result.results['0'].theta.mean(), 0, 1)
    assert_almost_equal(result.results['0'].theta.var(), 1./p, 1)

def test_glm_ar():
    n, p, q = 100, 80, 10
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    result = GLM(X).fit(Y)
    assert_true(len(result.labels) == n)
    assert_true(len(result.results.keys()) > 1)
    tmp = sum([result.results[key].theta.shape[1] 
               for key in result.results.keys()])
    assert_true(tmp == n)

def test_Tcontrast():
    n, p, q = 100, 80, 10
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    result = GLM(X).fit(Y)
    cval = np.hstack((1, np.ones(9)))
    z_vals = result.contrast(cval).z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)

def test_Fcontrast_1d():
    n, p, q = 100, 80, 10
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    result = GLM(X).fit(Y)
    cval = np.hstack((1, np.ones(9)))
    con = result.contrast(cval, contrast_type='F')
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)



if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
