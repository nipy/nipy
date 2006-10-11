import unittest
from neuroimaging.core.image.image import Image
from neuroimaging.algorithms.onesample import ImageOneSample
from neuroimaging.utils.tests.data import repository
from neuroimaging.data_io.formats.analyze import Analyze


class OneSampleTest(unittest.TestCase):


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
    unittest.main()
