import unittest, os

import numpy as N

from neuroimaging.utils.tests.data import repository
from neuroimaging.core.image import Image
from neuroimaging.algorithms.interpolation import ImageInterpolator

from neuroimaging.defines import pylab_def
PYLAB_DEF, pylab = pylab_def()
if PYLAB_DEF:
    from neuroimaging.ui.visualization import viewer, slices
    from neuroimaging.ui.visualization.montage import Montage

class MontageTest(unittest.TestCase):
    if PYLAB_DEF:
        def setUp(self):
            self.img = Image(repository.filename("rho.hdr"))
            self.interpolator = ImageInterpolator(self.img)
            r = self.img.grid.range()
            self.z = N.unique(r[0].flat); self.z.sort()
            self.y = N.unique(r[1].flat); self.y.sort()
            self.x = N.unique(r[2].flat); self.x.sort()

        def test_montage(self):
            rhoslices = {}
            vmax = float(self.img.readall().max()); vmin = float(self.img.readall().min())
            for i in range(5):
                for j in range(3):
                    if i*3 + j < 13:
                        cur_slice = slices.transversal(self.img.grid, z=self.z[i*3+j],
                                                       xlim=[-150,150.],
                                                       ylim=[-150.,150.])
                        rhoslices[j,i] = slices.DataSlicePlot(self.interpolator, cur_slice,
                                                              vmax=vmax, vmin=vmin,
                                                              colormap='spectral',
                                                              interpolation='nearest')
        
            m = Montage(slices=rhoslices, vmax=vmax, vmin=vmin)
            m.draw()
            pylab.savefig('image.png')
            os.remove('image.png')

def suite():
    return unittest.makeSuite(MontageTest)


if __name__ == '__main__':
    unittest.main()
