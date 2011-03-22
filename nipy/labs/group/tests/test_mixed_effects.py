# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from ..mixed_effects import em

import numpy as np
from numpy.testing import assert_array_almost_equal


def test_simple_model(): 
    size = 100
    X = np.zeros((size, 2))
    X[:,0] = 1.
    X[:,1] = range(size)
    err = .1
    sy = err*np.random.rand(size)
    e1 = sy*np.random.normal(size=size)
    e2 = .1*np.random.normal(size=size)
    y = X[:,0] + e1 + e2 
    vy = sy**2
    b, s2 = em(y, vy, X, niter=10)
    print b, s2
