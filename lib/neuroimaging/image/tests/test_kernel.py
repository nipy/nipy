import unittest

from neuroimaging.image import Image
from neuroimaging.image.kernel_smooth import LinearFilter
from neuroimaging.utils.tests.data import repository

from neuroimaging.defines import pylab_def
PYLAB_DEF, pylab = pylab_def()
if PYLAB_DEF:
    from neuroimaging.visualization import viewer
    from neuroimaging.fmri.pca import PCAmontage

class KernelTest(unittest.TestCase):
    def test_smooth(self):
        rho = Image("rho.hdr", repository)
        smoother = LinearFilter(rho.grid)

        if PYLAB_DEF:
            srho = smoother.smooth(rho)
            view = viewer.BoxViewer(rho)
            view.draw()

            sview = viewer.BoxViewer(srho)
            sview.m = view.m
            sview.M = view.M
            sview.draw()

def suite():
    suite = unittest.makeSuite(KernelTest)
    return suite

        
if __name__ == '__main__':
    unittest.main()
