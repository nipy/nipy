from numpy.testing import NumpyTest, NumpyTestCase

from neuroimaging.utils.test_decorators import slow, data

from neuroimaging.algorithms.onesample import ImageOneSample
from neuroimaging.core.api import load_image
from neuroimaging.utils.tests.data import repository


class test_OneSample(NumpyTestCase):


    def data_setUp(self):
        pass
    
    @slow
    @data
    def test_onesample1(self):
        im1 = load_image('FIAC/fiac3/fonc3/fsl/fmristat_run/contrasts/speaker/effect.hdr',
            repository)
        im2 = load_image('FIAC/fiac4/fonc3/fsl/fmristat_run/contrasts/speaker/effect.hdr',
            repository)
        im3 = load_image('FIAC/fiac5/fonc2/fsl/fmristat_run/contrasts/speaker/effect.hdr',
            repository)
        x = ImageOneSample([im1,im2,im3], clobber=True)
        x.fit()

from neuroimaging.utils.testutils import make_doctest_suite
test_suite = make_doctest_suite('neuroimaging.algorithms.onesample')

if __name__ == '__main__':
    NumpyTest.run()
