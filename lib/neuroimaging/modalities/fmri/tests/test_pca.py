import os

import numpy as N
from numpy.testing import NumpyTest, NumpyTestCase

from neuroimaging.utils.test_decorators import slow, data

from neuroimaging.modalities.fmri.fmri import fMRIImage
from neuroimaging.modalities.fmri.pca import PCA
from neuroimaging.core.api import Image
from neuroimaging.utils.tests.data import repository

from neuroimaging.defines import pylab_def
PYLAB_DEF, pylab = pylab_def()
if PYLAB_DEF:
    from neuroimaging.modalities.fmri.pca import PCAmontage

class test_PCA(NumpyTestCase):

    def setUp(self):
        pass

    def data_setUp(self):
        self.fmridata = fMRIImage("test_fmri.hdr", datasource=repository)


        frame = self.fmridata.frame(0)
        self.mask = Image(N.greater(frame.readall(), 500).astype(N.float64), grid=frame.grid)

class test_PCAMask(test_PCA):
    @slow
    @data
    def test_PCAmask(self):

        p = PCA(self.fmridata, mask=self.mask)
        p.fit()
        output = p.images(which=range(4))

class test_PCANoMask(test_PCA):
    @slow
    @data
    def test_PCA(self):

        p = PCA(self.fmridata)
        p.fit()
        output = p.images(which=range(4))

if PYLAB_DEF:

    class test_PCAMontageNoMask(test_PCA):
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

    class test_PCAMontageMask(test_PCA):
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


from neuroimaging.utils.testutils import make_doctest_suite
test_suite = make_doctest_suite('neuroimaging.modalities.fmri.pca')

        
if __name__ == '__main__':
    NumpyTest.main()
