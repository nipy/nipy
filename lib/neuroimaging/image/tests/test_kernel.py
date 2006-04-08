import unittest, os, scipy, glob, sets
import numpy as N
from neuroimaging.image import Image
import neuroimaging.image as image
import neuroimaging.image.kernel_smooth as kernel_smooth
from neuroimaging.visualization import viewer
import pylab

class KernelTest(unittest.TestCase):
    def test_smooth(self):
        rho = image.Image('http://kff.stanford.edu/~jtaylo/BrainSTAT/rho.img')
        smoother = kernel_smooth.LinearFilter(rho.grid)

        srho = smoother.smooth(rho)
        view = viewer.BoxViewer(rho)
        view.draw()

        sview = viewer.BoxViewer(srho)
        sview.m = view.m
        sview.M = view.M
        sview.draw()

##         pylab.show()

def suite():
    suite = unittest.makeSuite(KernelTest)
    return suite

        
if __name__ == '__main__':
    unittest.main()
