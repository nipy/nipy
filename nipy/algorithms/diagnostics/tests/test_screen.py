# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Testing diagnostic screen
"""

import numpy as np

import nipy as ni
from ..screens import screen
from ..timediff import time_slice_diffs
from ...utils.pca import pca
from ...utils.tests.test_pca import res2pos1

from nose.tools import (assert_true, assert_false, assert_equal, assert_raises)

from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)

from nipy.testing import funcfile, anatfile


def test_screen():
    img = ni.load_image(funcfile)
    res = screen(img)
    assert_equal(res['mean'].ndim, 3)
    assert_equal(res['pca'].ndim, 4)
    assert_equal(sorted(res.keys()),
                 ['max', 'mean', 'min',
                  'pca', 'pca_res',
                  'std', 'ts_res'])
    data = img.get_data()

    assert_array_equal(np.max(data, axis=-1), res['max'].get_data())
    assert_array_equal(np.mean(data, axis=-1), res['mean'].get_data())
    assert_array_equal(np.min(data, axis=-1), res['min'].get_data())
    assert_array_equal(np.std(data, axis=-1), res['std'].get_data())
    pca_res = pca(data, axis=-1, standardize=False, ncomp=10)
    # On windows, there seems to be some randomness in the PCA output vector
    # signs; this routine sets the basis vectors to have first value positive,
    # and therefore standardized the signs
    pca_res = res2pos1(pca_res)
    screen_pca_res = res2pos1(res['pca_res'])
    for key in pca_res:
        assert_almost_equal(pca_res[key], screen_pca_res[key])
    ts_res = time_slice_diffs(data)
    for key in ts_res:
        assert_array_equal(ts_res[key], res['ts_res'][key])
