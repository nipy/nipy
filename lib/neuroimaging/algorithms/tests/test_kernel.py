import unittest

from neuroimaging.algorithms.kernel_smooth import LinearFilter
from neuroimaging.core.image.image import Image
from neuroimaging.utils.tests.data import repository

from neuroimaging.utils.test_decorators import gui

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

def suite():
    suite = unittest.makeSuite(KernelTest)
    return suite

        
if __name__ == '__main__':
    unittest.main()
