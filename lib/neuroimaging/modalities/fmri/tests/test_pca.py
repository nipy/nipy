import unittest, os

import numpy as N

from neuroimaging.utils.test_decorators import slow, data

from neuroimaging.modalities.fmri import fMRIImage
from neuroimaging.modalities.fmri.pca import PCA
from neuroimaging.core.image.image import Image
from neuroimaging.utils.tests.data import repository

from neuroimaging.defines import pylab_def
PYLAB_DEF, pylab = pylab_def()
if PYLAB_DEF:
    from neuroimaging.modalities.fmri.pca import PCAmontage

class PCATest(unittest.TestCase):

    def setUp(self):
        pass

    def data_setUp(self):
        self.fmridata = fMRIImage("test_fmri.hdr", datasource=repository)


        frame = self.fmridata.frame(0)
        self.mask = Image(N.greater(frame.readall(), 500).astype(N.float64), grid=frame.grid)

class PCATestMask(PCATest):
    @slow
    @data
    def test_PCAmask(self):

        p = PCA(self.fmridata, mask=self.mask)
        p.fit()
        output = p.images(which=range(4))

class PCATestNoMask(PCATest):
    @slow
    @data
    def test_PCA(self):

        p = PCA(self.fmridata)
        p.fit()
        output = p.images(which=range(4))

if PYLAB_DEF:

    class PCATestMontageNoMask(PCATest):
        @slow
        @data
        def test_PCAmontage(self):
            p = PCAmontage(self.fmridata)
            p.fit()
            output = p.images(which=range(4))
            p.time_series()
            p.montage()
            pylab.savefig('image.png')
            os.remove('image.png')

    class PCATestMontageMask(PCATest):
        @slow
        @data
        def test_PCAmontage_nomask(self):
            p = PCAmontage(self.fmridata, mask=self.mask)
            p.fit()
            output = p.images(which=range(4))
            p.time_series()
            p.montage()
            pylab.savefig('image.png')
            os.remove('image.png')


def suite():
    suite = unittest.makeSuite([PCATestMask, PCATestNoMask,
                                PCATestMontageMask, PCATestMontageNoMask])
    return suite
        
if __name__ == '__main__':
    unittest.main()
