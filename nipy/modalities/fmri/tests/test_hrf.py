""" Testing hrf module
"""

import numpy as np

from scipy.stats import gamma

from nipy.modalities.fmri.hrf import (
    gamma_params,
    )

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nipy.testing import parametric


def test_gamma_params():
    t = np.linspace(0, 20, 5000)
    alpha = 3.2
    beta = 1.4
    gf = gamma(alpha, beta).pdf
    gt_t = gf(t)
    pk_i = np.argmax(gt_t)
    pk = gt_t[pk_i]
    at_hm_t = t[gt_t >= pk / 2.0]
    mn_t = np.min(at_hm_t)
    mx_t = np.max(at_hm_t)
    fwhm = mx_t - mn_t
    e_a, e_b, coef = gamma_params(pk, fwhm)
    print pk, fwhm, e_a, e_b
