from neuroimaging.statistics.pca import PCA, PCAmontage
from neuroimaging.fmri import fMRIImage
from neuroimaging.image import Image
import numpy as N

W = R.standard_normal

class PCATest(unittest.TestCase):

    def setUp(self):

        self.fmridata = fMRIImage('http://kff.stanford.edu/BrainSTAT/testdata/test_fmri.img')

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
