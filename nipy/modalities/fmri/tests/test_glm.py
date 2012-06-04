# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the glm utilities.
"""

import numpy as np
from nose.tools import assert_true
from numpy.testing import assert_almost_equal
from ..glm import GeneralLinearModel, data_scaling


def ols_glm(n=100, p=80, q=10):
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    glm = GeneralLinearModel(X)
    glm.fit(Y, 'ols')
    return glm, n, p, q


def ar1_glm(n=100, p=80, q=10):
    X, Y = np.random.randn(p, q), np.random.randn(p, n)
    glm = GeneralLinearModel(X)
    glm.fit(Y, 'ar1')
    return glm, n, p, q


def test_glm_ols():
    mulm, n, p, q = ols_glm()
    assert_true((mulm.labels_ == np.zeros(n)).all())
    assert_true(mulm.results_.keys() == [0.0])
    assert_true(mulm.results_[0.0].theta.shape == (q, n))
    assert_almost_equal(mulm.results_[0.0].theta.mean(), 0, 1)
    assert_almost_equal(mulm.results_[0.0].theta.var(), 1. / p, 1)


def test_glm_ar():
    mulm, n, p, q = ar1_glm()
    assert_true(len(mulm.labels_) == n)
    assert_true(len(mulm.results_.keys()) > 1)
    tmp = sum([mulm.results_[key].theta.shape[1]
               for key in mulm.results_.keys()])
    assert_true(tmp == n)


def test_Tcontrast():
    mulm, n, p, q = ar1_glm()
    cval = np.hstack((1, np.ones(9)))
    z_vals = mulm.contrast(cval).z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_Fcontrast_1d():
    mulm, n, p, q = ar1_glm()
    cval = np.hstack((1, np.ones(9)))
    con = mulm.contrast(cval, contrast_type='F')
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_Fcontrast_nd():
    mulm, n, p, q = ar1_glm()
    cval = np.eye(q)[:3]
    con = mulm.contrast(cval)
    assert_true(con.contrast_type == 'F')
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_Fcontrast_1d_old():
    mulm, n, p, q = ols_glm()
    cval = np.hstack((1, np.ones(9)))
    con = mulm.contrast(cval, contrast_type='F')
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_Fcontrast_nd_ols():
    mulm, n, p, q = ols_glm()
    cval = np.eye(q)[:3]
    con = mulm.contrast(cval)
    assert_true(con.contrast_type == 'F')
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_t_contrast_add():
    mulm, n, p, q = ols_glm()
    c1, c2 = np.eye(q)[0], np.eye(q)[1]
    con = mulm.contrast(c1) + mulm.contrast(c2)
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)


def test_F_contrast_add():
    mulm, n, p, q = ar1_glm()
    # first test with independent contrast
    c1, c2 = np.eye(q)[:2], np.eye(q)[2:4]
    con = mulm.contrast(c1) + mulm.contrast(c2)
    z_vals = con.z_score()
    assert_almost_equal(z_vals.mean(), 0, 0)
    assert_almost_equal(z_vals.std(), 1, 0)
    # first test with dependent contrast
    con1 = mulm.contrast(c1)
    con2 = mulm.contrast(c1) + mulm.contrast(c1)
    assert_almost_equal(con1.effect * 2, con2.effect)
    assert_almost_equal(con1.variance * 2, con2.variance)
    assert_almost_equal(con1.stat() * 2, con2.stat())


def test_t_contrast_mul():
    mulm, n, p, q = ar1_glm()
    con1 = mulm.contrast(np.eye(q)[0])
    con2 = con1 * 2
    assert_almost_equal(con1.z_score(), con2.z_score())
    assert_almost_equal(con1.effect * 2, con2.effect)


def test_F_contrast_mul():
    mulm, n, p, q = ar1_glm()
    con1 = mulm.contrast(np.eye(q)[:4])
    con2 = con1 * 2
    assert_almost_equal(con1.z_score(), con2.z_score())
    assert_almost_equal(con1.effect * 2, con2.effect)


def test_t_contrast_values():
    mulm, n, p, q = ar1_glm(n=1)
    cval = np.eye(q)[0]
    con = mulm.contrast(cval)
    t_ref = mulm.results_.values()[0].Tcontrast(cval).t
    assert_almost_equal(np.ravel(con.stat()), t_ref)


def test_F_contrast_calues():
    mulm, n, p, q = ar1_glm(n=1)
    cval = np.eye(q)[:3]
    con = mulm.contrast(cval)
    F_ref = mulm.results_.values()[0].Fcontrast(cval).F
    # Note that the values are not strictly equal,
    # this seems to be related to a bug in Mahalanobis
    assert_almost_equal(np.ravel(con.stat()), F_ref, 3)


def test_tmin():
    mulm, n, p, q = ar1_glm(n=1)
    #c1, c2 = np.eye(q)[0], np.eye(q)[1]
    #con1, con2 = mulm.contrast(c1), mulm.contrast(c2)
    c1, c2, c3 = np.eye(q)[0], np.eye(q)[1], np.eye(q)[2]
    t1, t2, t3 = mulm.contrast(c1).stat(), mulm.contrast(c2).stat(), \
        mulm.contrast(c3).stat()
    tmin = min(t1, t2, t3)
    con = mulm.contrast(np.eye(q)[:3])
    con.contrast_type = 'tmin'
    assert_true(con.stat() == tmin)


def test_scaling():
    """Test the scaling function"""
    shape = (400, 10)
    u = np.random.randn(*shape)
    mean = 100 * np.random.rand(shape[1])
    Y = u + mean
    Y, mean_ = data_scaling(Y)
    assert_almost_equal(Y.mean(0), 0)
    assert_almost_equal(mean_, mean, 0)
    assert_true(Y.std() > 1)
    

if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
