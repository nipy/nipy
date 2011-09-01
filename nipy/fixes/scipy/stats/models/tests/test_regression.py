# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test functions for models.regression
"""

import numpy as np
from numpy.random import standard_normal

import scipy.linalg as spl

from ..regression import OLSModel, ARModel, yule_walker

from nose.tools import assert_equal
from numpy.testing import assert_array_almost_equal


RNG = np.random.RandomState(20110901)
X = RNG.standard_normal((40,10))
Y = RNG.standard_normal((40,))


def test_OLS():
    model = OLSModel(design=X)
    results = model.fit(Y)
    assert_equal(results.df_resid, 30)


def test_AR():
    model = ARModel(design=X, rho=0.4)
    results = model.fit(Y)
    assert_equal(results.df_resid, 30)


def test_OLS_degenerate():
    Xd = X.copy()
    Xd[:,0] = Xd[:,1] + Xd[:,2]
    model = OLSModel(design=Xd)
    results = model.fit(Y)
    assert_equal(results.df_resid, 31)


def test_AR_degenerate():
    Xd = X.copy()
    Xd[:,0] = Xd[:,1] + Xd[:,2]
    model = ARModel(design=Xd, rho=0.9)
    results = model.fit(Y)
    assert_equal(results.df_resid, 31)


def test_yule_walker_R():
    # Test YW implementation against R results
    Y = np.array([1,3,4,5,8,9,10])
    N = len(Y)
    X = np.ones((N, 2))
    X[:,0] = np.arange(1,8)
    pX = spl.pinv(X)
    betas = np.dot(pX, Y)
    Yhat = Y - np.dot(X, betas)
    # R results obtained from:
    # >>> np.savetxt('yhat.csv', Yhat)
    # > yhat = read.table('yhat.csv')
    # > ar.yw(yhat$V1, aic=FALSE, order.max=2)
    def r_fudge(sigma, order):
        # Reverse fudge in ar.R calculation labeled as splus compatibility fix
        return sigma **2 * N / (N-order-1)
    rhos, sd = yule_walker(Yhat, 1, 'mle')
    assert_array_almost_equal(rhos, [-0.3004], 4)
    assert_array_almost_equal(r_fudge(sd, 1), 0.2534, 4)
    rhos, sd = yule_walker(Yhat, 2, 'mle')
    assert_array_almost_equal(rhos, [-0.5113, -0.7021], 4)
    assert_array_almost_equal(r_fudge(sd, 2), 0.1606, 4)
    rhos, sd = yule_walker(Yhat, 3, 'mle')
    assert_array_almost_equal(rhos, [-0.6737, -0.8204, -0.2313], 4)
    assert_array_almost_equal(r_fudge(sd, 3), 0.2027, 4)
