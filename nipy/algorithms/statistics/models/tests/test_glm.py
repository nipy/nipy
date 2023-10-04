# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test functions for models.GLM
"""

import numpy as np
import pytest

from .. import family
from ..glm import Model as GLM


@pytest.fixture
def x_y():
    rng = np.random.RandomState(20110928)
    X = rng.standard_normal((40,10))
    Y = rng.standard_normal((40,))
    Y = np.greater(Y, 0)
    return {'X': X, 'Y': Y}


def test_Logistic(x_y):
    X = x_y['X']
    Y = x_y['Y']
    cmodel = GLM(design=X, family=family.Binomial())
    results = cmodel.fit(Y)
    assert results.df_resid == 30


def test_cont(x_y):
    # Test continue function works as expected
    X = x_y['X']
    Y = x_y['Y']
    cmodel = GLM(design=X, family=family.Binomial())
    cmodel.fit(Y)
    assert cmodel.cont(0)
    assert not cmodel.cont(np.inf)


def test_Logisticdegenerate(x_y):
    X = x_y['X'].copy()
    X[:,0] = X[:,1] + X[:,2]
    Y = x_y['Y']
    cmodel = GLM(design=X, family=family.Binomial())
    results = cmodel.fit(Y)
    assert results.df_resid == 31
