import os

import numpy as N
from neuroimaging.testing import *



from neuroimaging.modalities.fmri.api import FmriImage, fromimage
from neuroimaging.modalities.fmri.pca import PCA
from neuroimaging.core.api import Image, load_image
from neuroimaging.testing import funcfile


class test_PCA(TestCase):

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

class test_PCAMontageNoMask(test_PCA):
    @slow
    @data
    def test_PCAmontage(self):
        from neuroimaging.modalities.fmri.pca import PCAmontage
        from pylab import savefig
        p = PCAmontage(self.fmridata)
        p.fit()
        output = p.images(which=range(4))
        p.time_series()
        p.montage()
        savefig('image.png')
        os.remove('image.png')

class test_PCAMontageMask(test_PCA):
    @slow
    @data
    def test_PCAmontage_nomask(self):
        from neuroimaging.modalities.fmri.pca import PCAmontage
        from pylab import savefig
        p = PCAmontage(self.fmridata, mask=self.mask)
        p.fit()
        output = p.images(which=range(4))
        p.time_series()
        p.montage()
        savefig('image.png')
        os.remove('image.png')




        
if __name__ == '__main__':
    run_module_suite()
