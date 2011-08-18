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


W = standard_normal

def test_OLS():
    X = W((40,10))
    Y = W((40,))
    model = OLSModel(design=X)
    results = model.fit(Y)
    assert_equal(results.df_resid, 30)


def test_AR():
    X = W((40,10))
    Y = W((40,))
    model = ARModel(design=X, rho=0.4)
    results = model.fit(Y)
    assert_equal(results.df_resid, 30)


def test_OLS_degenerate():
    X = W((40,10))
    X[:,0] = X[:,1] + X[:,2]
    Y = W((40,))
    model = OLSModel(design=X)
    results = model.fit(Y)
    assert_equal(results.df_resid, 31)


def test_AR_degenerate():
    X = W((40,10))
    X[:,0] = X[:,1] + X[:,2]
    Y = W((40,))
    model = ARModel(design=X, rho=0.9)
    results = model.fit(Y)
    assert_equal(results.df_resid, 31)


def test_yule_walker():
    # Test YW implementation against R results
    Y = np.array([1,3,4,5,8,9,10])
    N = len(Y)
    X = np.ones((N, 2))
    X[:,0] = np.arange(1,8)
    pX = spl.pinv(X)
    betas = pX.dot(Y)
    Yhat = Y - X.dot(betas)
    # R results from saving Yhat, then:
    # ar.yw(yhat$V1, aic=FALSE, order.max=2)
    rhos, sd = yule_walker(Yhat, 1, 'mle')
    assert_array_almost_equal(rhos, [-0.3004], 4)
    rhos, sd = yule_walker(Yhat, 2, 'mle')
    assert_array_almost_equal(rhos, [-0.5113, -0.7021], 4)
    rhos, sd = yule_walker(Yhat, 3, 'mle')
    assert_array_almost_equal(rhos, [-0.6737, -0.8204, -0.2313], 4)

