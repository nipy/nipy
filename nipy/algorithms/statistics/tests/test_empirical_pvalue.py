# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the empirical null estimator.
"""
import warnings

import numpy as np

from ..empirical_pvalue import \
    NormalEmpiricalNull, smoothed_histogram_from_samples

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
    #
    # make the tests
    efdr = NormalEmpiricalNull(x)
    np.testing.assert_array_less(efdr.fdr(3.0), 0.2)
    np.testing.assert_array_less(-efdr.threshold(alpha=0.05), -2.8)
    np.testing.assert_array_less(-efdr.uncorrected_threshold(alpha=0.001), -2.5)

def test_smooth_histo():
   """
   test smoothed histogram geenration
   """
   n=100
   x = np.random.randn(n)
   h, c = smoothed_histogram_from_samples(x, normalized=True)
   thh = 1./np.sqrt(2*np.pi)
   hm = h.max()
   #print hm, thh
   assert np.absolute(hm-thh)<0.15

if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
