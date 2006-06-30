import unittest

import numpy as N
import pylab

from neuroimaging.fmri import fMRIImage
from neuroimaging.fmri.pca import PCA, PCAmontage
from neuroimaging.image import Image
from neuroimaging.tests.data import repository

class PCATest(unittest.TestCase):

    def setUp(self):

        self.fmridata = fMRIImage('http://kff.stanford.edu/BrainSTAT/testdata/test_fmri.img')
        #self.fmridata = fMRIImage("test_fmri.img", repository)


        frame = fmridata.frame(0)
        self.mask = Image(N.greater(frame.readall(), 500).astype(N.Float), grid=frame.grid)

    def test_PCAmask(self):

        p = PCA(fmridata, mask=mask)
        p.fit()
        output = p.images(which=range(4))

    def test_PCA(self):

        p = PCA(fmridata)
        p.fit()
        output = p.images(which=range(4))

    def test_PCAmontage(self):
        p = PCA(fmridata, mask=mask)
        p.fit()
        output = p.images(which=range(4))
        p.time_series()
        p.montage()
        pylab.show()

    def test_PCAmontage_nomask(self):
        p = PCA(fmridata, mask=mask)
        p.fit()
        output = p.images(which=range(4))
        p.time_series()
        p.montage()
        pylab.show()


def suite():
    suite = unittest.makeSuite(PCATest)
    return suite
        

if __name__ == '__main__':
    unittest.main()
