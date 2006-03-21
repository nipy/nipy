import unittest, os, scipy, glob, sets
import numpy as N
from neuroimaging.image import Image, interpolation
from neuroimaging.visualization import viewer, slices
from neuroimaging.reference import slices as rslices
import pylab

class VisualizationTest(unittest.TestCase):

    def setUp(self):
        url = 'http://kff.stanford.edu/~jtaylo/BrainSTAT/rho.img'
        self.img = Image(url)
        self.interpolator = interpolation.ImageInterpolator(self.img)
        self.m = float(self.img.readall().min())
        self.M = float(self.img.readall().max())

    def test_view(self):
        view = viewer.BoxViewer(self.img, z_pix=80.)
        view.draw()
        pylab.show()

    def test_transversal_slice(self):

        _slice = slices.transversal(self.img, z=0., xlim=[-20,20.], ylim=[0,40.])

        x = slices.PylabDataSlice(self.interpolator,
                                  _slice,
                                  vmax=self.M,
                                  vmin=self.m,
                                  colormap='spectral',
                                  interpolation='nearest')
        x.height = 0.8
        x.width = 0.8

        pylab.figure(figsize=(3,3))
        x.getaxes()


        pylab.imshow(x.RGBA(), origin=x.origin)
        pylab.show()
      
    def test_transversal_slice2(self):

        x = slices.PylabTransversal(self.img, y=3., xlim=[-49.,35.])
        x.height = 0.8
        x.width = 0.8
        pylab.figure(figsize=(3,3))
        x.getaxes()

        pylab.imshow(x.RGBA(), origin=x.origin)
        pylab.show()
      


if __name__ == '__main__':
    unittest.main()
