# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the glm utilities.
"""

import numpy as np
from nose.tools import assert_true
from numpy.testing import assert_almost_equal
from ..glm import glm_fit, GLMResults, Contrast

def ols_glm(n=100, p=80, q=10):
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    return glm_fit(X, Y, 'ols'), n, p, q
    
def ar1_glm(n=100, p=80, q=10):
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    return glm_fit(X, Y), n, p, q
    

def test_glm_ols():
    result, n, p, q = ols_glm () 
    assert_true((result.labels == np.zeros(n)).all())
    assert_true(result.results.keys() == [0.0])
    assert_true(result.results[0.0].theta.shape == (q, n))
    assert_almost_equal(result.results[0.0].theta.mean(), 0, 1)
    assert_almost_equal(result.results[0.0].theta.var(), 1. / p, 1)

def test_glm_ar():
    result, n, p, q = ar1_glm()
    assert_true(len(result.labels) == n)
    assert_true(len(result.results.keys()) > 1)
    tmp = sum([result.results[key].theta.shape[1] 
               for key in result.results.keys()])
    assert_true(tmp == n)

def test_Tcontrast():
    result, n, p, q = ar1_glm()
    cval = np.hstack((1, np.ones(9)))
    z_vals = result.contrast(cval).z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)

def test_Fcontrast_1d():
    result, n, p, q = ar1_glm()
    cval = np.hstack((1, np.ones(9)))
    con = result.contrast(cval, contrast_type='F')
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)

def test_Fcontrast_nd():
    result, n, p, q = ar1_glm()
    cval = np.eye(q)[:3]
    con = result.contrast(cval)
    assert_true(con.contrast_type == 'F')
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)

def test_Fcontrast_1d_old():
    result, n, p, q = ols_glm()
    cval = np.hstack((1, np.ones(9)))
    con = result.contrast(cval, contrast_type='F')
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)

def test_Fcontrast_nd_ols():
    result, n, p, q = ols_glm()
    cval = np.eye(q)[:3]
    con = result.contrast(cval)
    assert_true(con.contrast_type == 'F')
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)

def test_t_contrast_add():
    result, n, p, q = ols_glm()
    c1, c2 = np.eye(q)[0], np.eye(q)[1]
    con = result.contrast(c1) + result.contrast(c2)
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)

def test_F_contrast_add():
    result, n, p, q = ar1_glm()
    # first test with independent contrast
    c1, c2 = np.eye(q)[:2], np.eye(q)[2:4]
    con = result.contrast(c1) + result.contrast(c2)
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)
    # first test with dependent contrast
    con1 = result.contrast(c1)
    con2 = result.contrast(c1) + result.contrast(c1)
    assert_almost_equal(con1.effect * 2, con2.effect)
    assert_almost_equal(con1.variance * 2, con2.variance)
    assert_almost_equal(con1.stat() * 2, con2.stat())

def test_t_contrast_mul():
    result, n, p, q = ar1_glm()
    con1 = result.contrast(np.eye(q)[0])
    con2 = con1 * 2
    assert_almost_equal(con1.z_score(), con2.z_score())
    assert_almost_equal(con1.effect * 2, con2.effect)

def test_F_contrast_mul():
    result, n, p, q = ar1_glm()
    con1 = result.contrast(np.eye(q)[:4])
    con2 = con1 * 2
    assert_almost_equal(con1.z_score(), con2.z_score())
    assert_almost_equal(con1.effect * 2, con2.effect)

def test_t_contrast_values():
    result, n, p, q = ar1_glm(n=1)
    cval = np.eye(q)[0]
    con = result.contrast(cval)
    t_ref = result.results.values()[0].Tcontrast(cval).t
    assert_almost_equal(np.ravel(con.stat()), t_ref)

def test_F_contrast_calues():
    result, n, p, q = ar1_glm(n=1)
    cval = np.eye(q)[:3]
    con = result.contrast(cval)
    F_ref = result.results.values()[0].Fcontrast(cval).F
    # Note that the values are not strictly equal, 
    # this seems to be related to a bug in Mahalanobis
    assert_almost_equal(np.ravel(con.stat()), F_ref, 3)

def test_tmin():
    result, n, p, q = ar1_glm(n=1)
    #c1, c2 = np.eye(q)[0], np.eye(q)[1]
    #con1, con2 = result.contrast(c1), result.contrast(c2)
    c1, c2, c3 = np.eye(q)[0], np.eye(q)[1], np.eye(q)[2]
    t1, t2, t3 = result.contrast(c1).stat(), result.contrast(c2).stat(), \
        result.contrast(c3).stat()
    tmin = min(t1, t2, t3)
    con = result.contrast(np.eye(q)[:3])
    con.contrast_type = 'tmin'
    assert_true(con.stat() == tmin)


if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
