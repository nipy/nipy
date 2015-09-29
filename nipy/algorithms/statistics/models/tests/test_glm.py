# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test functions for models.GLM
"""
from __future__ import absolute_import

import numpy as np

from .. import family
from ..glm import Model as GLM

from nose.tools import assert_equal, assert_true, assert_false

VARS = {}

def setup():
    rng = np.random.RandomState(20110928)
    VARS['X'] = rng.standard_normal((40,10))
    Y = rng.standard_normal((40,))
    VARS['Y'] = np.greater(Y, 0)


def test_Logistic():
    X = VARS['X']
    Y = VARS['Y']
    cmodel = GLM(design=X, family=family.Binomial())
    results = cmodel.fit(Y)
    assert_equal(results.df_resid, 30)


def test_cont():
    # Test continue function works as expected
    X = VARS['X']
    Y = VARS['Y']
    cmodel = GLM(design=X, family=family.Binomial())
    cmodel.fit(Y)
    assert_true(cmodel.cont(0))
    assert_false(cmodel.cont(np.inf))


def test_Logisticdegenerate():
    X = VARS['X'].copy()
    X[:,0] = X[:,1] + X[:,2]
    Y = VARS['Y']
    cmodel = GLM(design=X, family=family.Binomial())
    results = cmodel.fit(Y)
    assert_equal(results.df_resid, 31)
