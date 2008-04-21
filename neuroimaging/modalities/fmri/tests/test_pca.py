import os

import numpy as N
from numpy.testing import NumpyTest, NumpyTestCase

from neuroimaging.utils.test_decorators import slow, data

from neuroimaging.modalities.fmri.api import FmriImage, fromimage
from neuroimaging.modalities.fmri.pca import PCA
from neuroimaging.core.api import Image, load_image
from neuroimaging.testing import funcfile

## from neuroimaging.defines import pylab_def
## PYLAB_DEF, pylab = pylab_def()
## if PYLAB_DEF:
##     from neuroimaging.modalities.fmri.pca import PCAmontage

class test_PCA(NumpyTestCase):

    def setUp(self):
        self.img = load_image(funcfile)
        self.fmridata = fromimage(self.img)

        frame = self.fmridata[0]
        self.mask = Image(N.greater(N.asarray(frame), 500).astype(N.float64),
                          frame.grid)

class test_PCAMask(test_PCA):
#    @slow
    def test_PCAmask(self):

        p = PCA(self.fmridata, self.mask)
        p.fit()
        output = p.images(which=range(4))

class test_PCANoMask(test_PCA):
    @slow
    @data
    def test_PCA(self):

        p = PCA(self.fmridata)
        p.fit()
        output = p.images(which=range(4))

## if PYLAB_DEF:

##     class test_PCAMontageNoMask(test_PCA):
##         @slow
##         @data
##         def test_PCAmontage(self):
##             p = PCAmontage(self.fmridata)
##             p.fit()
##             output = p.images(which=range(4))
##             p.time_series()
##             p.montage()
##             pylab.savefig('image.png')
##             os.remove('image.png')

##     class test_PCAMontageMask(test_PCA):
##         @slow
##         @data
##         def test_PCAmontage_nomask(self):
##             p = PCAmontage(self.fmridata, mask=self.mask)
##             p.fit()
##             output = p.images(which=range(4))
##             p.time_series()
##             p.montage()
##             pylab.savefig('image.png')
##             os.remove('image.png')


from neuroimaging.utils.testutils import make_doctest_suite
test_suite = make_doctest_suite('neuroimaging.modalities.fmri.pca')

        
if __name__ == '__main__':
    NumpyTest.main()
