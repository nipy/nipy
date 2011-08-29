# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test functions for models.GLM
"""

import numpy as np

from .. import family
from ..glm import Model as GLM

from nose.tools import assert_equal

W = np.random.standard_normal


def test_Logistic():
    X = W((40,10))
    Y = np.greater(W((40,)), 0)
    cmodel = GLM(design=X, family=family.Binomial())
    results = cmodel.fit(Y)
    assert_equal(results.df_resid, 30)


def test_Logisticdegenerate():
    X = W((40,10))
    X[:,0] = X[:,1] + X[:,2]
    Y = np.greater(W((40,)), 0)
    cmodel = GLM(design=X, family=family.Binomial())
    results = cmodel.fit(Y)
    assert_equal(results.df_resid, 31)
