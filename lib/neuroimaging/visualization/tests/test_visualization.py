import unittest

import pylab

from neuroimaging.tests.data import repository
from neuroimaging.image import Image
from neuroimaging.image.interpolation import ImageInterpolator
from neuroimaging.visualization import viewer, slices

class VisualizationTest(unittest.TestCase):

    def setUp(self):
        self.img = Image(repository.filename("rho.img"))

    def test_view(self):
        view = viewer.BoxViewer(self.img, z_pix=80.)
        view.draw()
        pylab.show()

    def test_transversal_slice(self):
        interpolator = ImageInterpolator(self.img)
        vmin = float(self.img.readall().min())
        vmax = float(self.img.readall().max())
        _slice = slices.transversal(self.img, z=0.,
          xlim=[-20,20.], ylim=[0,40.])
        x = slices.DataSlicePlot(interpolator, _slice, vmax=vmax, vmin=vmin,
          colormap='spectral', interpolation='nearest')
        x.width = 0.8; x.height = 0.8
        pylab.figure(figsize=(3,3))
        x.getaxes()
        pylab.imshow(x.RGBA(), origin=x.origin)
        pylab.show()
      
    def test_transversal_slice2(self):
        x = slices.TransversalPlot(self.img, y=3., xlim=[-49.,35.])
        x.width = 0.8; x.height = 0.8
        pylab.figure(figsize=(3,3))
        x.getaxes()
        pylab.imshow(x.RGBA(), origin=x.origin)
        pylab.show()

    def test_arrayview(self):
        from neuroimaging.visualization import arrayview
        arrayview.arrayview(self.img.readall())

def suite():
    return unittest.makeSuite(VisualizationTest)


if __name__ == '__main__':
    unittest.main()
