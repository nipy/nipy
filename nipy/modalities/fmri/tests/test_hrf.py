""" Testing hrf module
"""
from __future__ import absolute_import

from os.path import dirname, join as pjoin

import numpy as np

from scipy.stats import gamma
import scipy.io as sio

from ..hrf import (
    gamma_params,
    gamma_expr,
    lambdify_t,
    spm_hrf_compat,
    spmt,
    dspmt,
    ddspmt,
    )

from nose.tools import assert_raises
from numpy.testing import assert_almost_equal


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
    assert_almost_equal(L1t , L2t*sf)


def test_spm_hrf():
    # Regression tests for spm hrf, time derivative and dispersion derivative
    # Check that absolute values don't change (much) with different dt, and that
    # max values are roughly the same and in the same place in time
    for dt in 0.1, 0.01, 0.001:
        t_vec = np.arange(0, 32, dt)
        hrf = spmt(t_vec)
        assert_almost_equal(np.max(hrf), 0.21053, 5)
        assert_almost_equal(t_vec[np.argmax(hrf)], 5, 2)
        dhrf = dspmt(t_vec)
        assert_almost_equal(np.max(dhrf), 0.08, 3)
        assert_almost_equal(t_vec[np.argmax(dhrf)], 3.3, 1)
        dhrf = ddspmt(t_vec)
        assert_almost_equal(np.max(dhrf), 0.10, 2)
        assert_almost_equal(t_vec[np.argmax(dhrf)], 5.7, 1)
    # Test reversed time vector to check that order of time values does not
    # affect result
    rt_vec = np.arange(0, 32, 0.01)
    rhrf = spmt(rt_vec)
    assert_almost_equal(np.max(rhrf), 0.21053, 5)
    assert_almost_equal(t_vec[np.argmax(hrf)], 5, 2)


def test_spm_hrf_octave():
    # Test SPM hrf against output from SPM code running in Octave
    my_path = dirname(__file__)
    hrfs_path = pjoin(my_path, 'spm_hrfs.mat')
    # mat file resulting from make_hrfs.m
    hrfs_mat = sio.loadmat(hrfs_path, squeeze_me=True)
    params = hrfs_mat['params']
    hrfs = hrfs_mat['hrfs']
    for i, pvec in enumerate(params):
        dt, ppk, upk, pdsp, udsp, rat = pvec
        t_vec = np.arange(0, 32.1, dt)
        our_hrf = spm_hrf_compat(t_vec,
                                 peak_delay=ppk,
                                 peak_disp=pdsp,
                                 under_delay=upk,
                                 under_disp=udsp,
                                 p_u_ratio=rat)
        # Normalize integral to match SPM
        assert_almost_equal(our_hrf, hrfs[i])
    # Test basis functions
    # mat file resulting from get_td_dd.m
    bases_path = pjoin(my_path, 'spm_bases.mat')
    bases_mat = sio.loadmat(bases_path, squeeze_me=True)
    dt = bases_mat['dt']
    t_vec = np.arange(0, 32 + dt, dt)
    # SPM function divides by sum of values - revert with dt
    assert_almost_equal(spmt(t_vec), bases_mat['hrf'] / dt, 4)
    assert_almost_equal(dspmt(t_vec), bases_mat['dhrf'] / dt, 4)
    assert_almost_equal(ddspmt(t_vec), bases_mat['ddhrf'] / dt, 4)


def test_spm_hrf_errors():
    t_vec = np.arange(0, 32)
    # All 1s is fine
    res = spm_hrf_compat(t_vec, 1, 1, 1, 1)
    # 0 or negative raise error for other args
    args = [0]
    for i in range(4):
        assert_raises(ValueError, spm_hrf_compat, t_vec, *args)
        args[-1] = -1
        assert_raises(ValueError, spm_hrf_compat, t_vec, *args)
        args[-1] = 1
        args.append(0)
