import unittest
import numpy as N

from neuroimaging.algorithms.utils import sigma2fwhm, fwhm2sigma

class utilTest(unittest.TestCase):
   

    def test_sigma_fwhm(self):
        """
        ensure that fwhm2sigma and sigma2fwhm are inverses of each other        
        """
        fwhm = N.arange(1.0, 5.0, 0.1)
        sigma = N.arange(1.0, 5.0, 0.1)
        N.testing.assert_almost_equal(sigma2fwhm(fwhm2sigma(fwhm)), fwhm)
        N.testing.assert_almost_equal(fwhm2sigma(sigma2fwhm(sigma)), sigma)
        

if __name__ == '__main__':
    unittest.main()
