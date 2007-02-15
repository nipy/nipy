import unittest

import numpy as N

from neuroimaging.algorithms.kernel_smooth import LinearFilter
from neuroimaging.core.api import Image
from neuroimaging.utils.tests.data import repository

from neuroimaging.utils.test_decorators import gui
from neuroimaging.algorithms.kernel_smooth import sigma2fwhm, fwhm2sigma

from neuroimaging.defines import pylab_def


PYLAB_DEF, pylab = pylab_def()
if PYLAB_DEF:
    from neuroimaging.ui.visualization.viewer import BoxViewer
    from neuroimaging.modalities.fmri.pca import PCAmontage

class KernelTest(unittest.TestCase):
    @gui
    def test_smooth(self):
        rho = Image("rho.hdr", repository)
        smoother = LinearFilter(rho.grid)

        if PYLAB_DEF:
            srho = smoother.smooth(rho)
            view = BoxViewer(rho)
            view.draw()

            sview = BoxViewer(srho)
            sview.m = view.m
            sview.M = view.M
            sview.draw()
            pylab.show()

class SigmaFWHMTest(unittest.TestCase):
    def test_sigma_fwhm(self):
        """
        ensure that fwhm2sigma and sigma2fwhm are inverses of each other        
        """
        fwhm = N.arange(1.0, 5.0, 0.1)
        sigma = N.arange(1.0, 5.0, 0.1)
        N.testing.assert_almost_equal(sigma2fwhm(fwhm2sigma(fwhm)), fwhm)
        N.testing.assert_almost_equal(fwhm2sigma(sigma2fwhm(sigma)), sigma)


def suite():
    suite = unittest.makeSuite(KernelTest)
    return suite

        
if __name__ == '__main__':
    unittest.main()
