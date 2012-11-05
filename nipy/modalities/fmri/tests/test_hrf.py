""" Testing hrf module
"""

import numpy as np

from scipy.stats import gamma

from ..hrf import (
    gamma_params,
    gamma_expr,
    lambdify_t,
    )

from numpy.testing import assert_array_almost_equal


def test_gamma():
    t = np.linspace(0, 30, 5000)
    # make up some numbers
    pk_t = 5.0
    fwhm = 6.0
    # get the estimated parameters
    shape, scale, coef = gamma_params(pk_t, fwhm)
    # get distribution function
    g_exp = gamma_expr(pk_t, fwhm)
    # make matching standard distribution
    gf = gamma(shape, scale=scale).pdf
    # get values
    L1t = gf(t)
    L2t = lambdify_t(g_exp)(t)
    # they are the same bar a scaling factor
    nz = np.abs(L1t) > 1e-15
    sf = np.mean(L1t[nz] / L2t[nz])
    assert_array_almost_equal(L1t , L2t*sf)
