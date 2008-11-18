from neuroimaging.testing import *

from neuroimaging.algorithms.onesample import ImageOneSample
from neuroimaging.core.api import load_image
from neuroimaging.utils.tests.data import repository

class test_OneSample(TestCase):
    @dec.knownfailure
    @dec.slow
    @dec.data
    def test_onesample1(self):
        # FIXME: When we replace nipy's datasource with numpy's
        # datasource, remove the string casting.  _fullpath returns a
        # 'path' object.
        fp1 = repository._fullpath('FIAC/fiac3/fonc3/fsl/fmristat_run/contrasts/speaker/effect.nii.gz')
        fp1 = str(fp1)
        im1 = load_image(fp1)

        fp2 = repository._fullpath('FIAC/fiac4/fonc3/fsl/fmristat_run/contrasts/speaker/effect.nii.gz')
        fp2 = str(fp2)
        im2 = load_image(fp2)

        fp3 = repository._fullpath('FIAC/fiac5/fonc2/fsl/fmristat_run/contrasts/speaker/effect.nii.gz')
        fp3 = str(fp3)
        im3 = load_image(fp3)

        # FIXME: ImageSequenceIterator is not defined.
        # ImageOneSample.__init__ fails.
        #   File "/Users/cburns/src/nipy-trunk/neuroimaging/algorithms/onesample.py", line 68, in __init__
        # self.iterator = ImageSequenceIterator(input)
        # NameError: global name 'ImageSequenceIterator' is not defined
        x = ImageOneSample([im1,im2,im3], clobber=True)
        x.fit()
