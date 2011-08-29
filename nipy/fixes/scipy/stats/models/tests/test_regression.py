# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test functions for models.regression
"""

import numpy as np

from ..regression import OLSModel, ARModel

from nose.tools import assert_equal

W = np.random.standard_normal


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


def test_OLSdegenerate():
    X = W((40,10))
    X[:,0] = X[:,1] + X[:,2]
    Y = W((40,))
    model = OLSModel(design=X)
    results = model.fit(Y)
    assert_equal(results.df_resid, 31)


def test_ARdegenerate():
    X = W((40,10))
    X[:,0] = X[:,1] + X[:,2]
    Y = W((40,))
    model = ARModel(design=X, rho=0.9)
    results = model.fit(Y)
    assert_equal(results.df_resid, 31)
