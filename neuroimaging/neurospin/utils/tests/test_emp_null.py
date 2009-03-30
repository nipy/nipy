"""
Test the empirical null estimator.
"""
import numpy as np

from neuroimaging.neurospin.utils.emp_null import ENN

def test_efdr():
    # generate the data
    n = 100000
    x = np.random.randn(n)
    x[:3000] += 3
    #
    # make the tests
    efdr = ENN(x)
    np.testing.assert_array_less(efdr.fdr(3.0), 0.15)
    np.testing.assert_array_less(-efdr.threshold(alpha=0.05), -3)
    np.testing.assert_array_less(-efdr.uncorrected_threshold(alpha=0.001), -3)

