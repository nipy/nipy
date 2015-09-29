# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the empirical null estimator.
"""
from __future__ import absolute_import
import warnings
import numpy as np
from nose.tools import assert_true


from ..empirical_pvalue import (
    NormalEmpiricalNull, smoothed_histogram_from_samples, fdr, fdr_threshold, 
    gaussian_fdr_threshold, gaussian_fdr)

def setup():
    # Suppress warnings during tests to reduce noise
    warnings.simplefilter("ignore")

def teardown():
    # Clear list of warning filters
    warnings.resetwarnings()

def test_efdr():
    # generate the data
    n = 100000
    x = np.random.randn(n)
    x[:3000] += 3
    # make the tests
    efdr = NormalEmpiricalNull(x)
    np.testing.assert_array_less(efdr.fdr(3.0), 0.2)
    np.testing.assert_array_less(-efdr.threshold(alpha=0.05), -2.8)
    np.testing.assert_array_less(-efdr.uncorrected_threshold(alpha=0.001), -2.5)

def test_smooth_histo():
   n = 100
   x = np.random.randn(n)
   h, c = smoothed_histogram_from_samples(x, normalized=True)
   thh = 1. / np.sqrt(2 * np.pi)
   hm = h.max()
   assert_true(np.absolute(hm - thh) < 0.15)

def test_fdr_pos():
    # test with some significant values
    np.random.seed([1])
    x = np.random.rand(100)
    x[:10] *= (.05 / 10)
    q = fdr(x)
    assert_true((q[:10] < .05).all())
    pc = fdr_threshold(x)
    assert_true((pc > .0025) & (pc < .1))

def test_fdr_neg():
    # test without some significant values
    np.random.seed([1])
    x = np.random.rand(100) * .8 + .2
    q =fdr(x)
    assert_true((q > .05).all())
    pc = fdr_threshold(x)
    assert_true(pc == .05 / 100)

def test_gaussian_fdr():
    # Test that fdr works on Gaussian data
    np.random.seed([2])
    x = np.random.randn(100) * 2
    fdr = gaussian_fdr(x)
    assert_true(fdr.min() < .05)
    assert_true(fdr.max() > .99)

def test_gaussian_fdr_threshold():
    np.random.seed([2])
    x = np.random.randn(100) * 2
    ac = gaussian_fdr_threshold(x)
    assert_true(ac > 2.0)
    assert_true(ac < 4.0) 
    assert_true(ac > gaussian_fdr_threshold(x, alpha=.1))

if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
