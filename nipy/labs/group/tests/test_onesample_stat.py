# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
from numpy.testing import assert_array_equal

from .. import onesample


def test_onesample_stat():
    dx, dy, dz = 3, 4, 2
    nvox = dx*dy*dz
    nsub = 12
    # Make surrogate data
    aux = np.arange(nvox)
    x = np.reshape(aux.repeat(nsub), [dx, dy, dz, nsub])
    # Gold standard
    y_target = np.inf * np.ones(nvox)
    y_target[0] = 0.0
    # Test: input C-contiguous, data owner, axis=3
    y = onesample.stat(x, axis=3).reshape(nvox)
    assert_array_equal(y, y_target)
    # Test: input F-contiguous, not owner, axis=0
    y = onesample.stat(x.T, axis=0).reshape(nvox)
    assert_array_equal(y, y_target)
    # Test: input C-contiguous, data owner, axis=0
    xT = x.T.copy()
    y = onesample.stat(xT, axis=0).reshape(nvox)
    assert_array_equal(y, y_target)
