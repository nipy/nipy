from numpy.testing import NumpyTest, NumpyTestCase

from neuroimaging.utils.test_decorators import slow, data

from neuroimaging.algorithms.onesample import ImageOneSample
from neuroimaging.core.api import Image
from neuroimaging.data_io.formats.analyze import Analyze
from neuroimaging.utils.tests.data import repository


class test_OneSample(NumpyTestCase):


    def data_setUp(self):
        pass
    
    @slow
    @data
    def test_onesample1(self):
        im1 = Image('FIAC/fiac3/fonc3/fsl/fmristat_run/contrasts/speaker/effect.hdr',
            repository, format=Analyze)
        im2 = Image('FIAC/fiac4/fonc3/fsl/fmristat_run/contrasts/speaker/effect.hdr',
            repository, format=Analyze)
        im3 = Image('FIAC/fiac5/fonc2/fsl/fmristat_run/contrasts/speaker/effect.hdr',
            repository, format=Analyze)
        x = ImageOneSample([im1,im2,im3], clobber=True)
        x.fit()

if __name__ == '__main__':
    NumpyTest.run()
