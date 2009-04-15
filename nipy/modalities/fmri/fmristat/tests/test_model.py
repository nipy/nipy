import os
import warnings
from shutil import rmtree
from tempfile import mkstemp, mkdtemp

from nipy.testing import *

import nipy.modalities.fmri.fmristat.model as model
from nipy.modalities.fmri.api import fromimage
from nipy.io.api import load_image

from nipy.modalities.fmri.protocol import Formula, \
    ExperimentalQuantitative
from nipy.fixes.scipy.stats.models.contrast import Contrast

def setup():
    # Suppress warnings during tests to reduce noise
    warnings.simplefilter("ignore")

def teardown():
    # Clear list of warning filters
    warnings.resetwarnings()


class test_fMRIstat_model(TestCase):

    def setUp(self):
        # Using mkstemp instead of NamedTemporaryFile.  MS Windows
        # cannot reopen files created with NamedTemporaryFile.
        _, self.ar1 = mkstemp(prefix='ar1_', suffix='.nii')
        _, self.resid_OLS = mkstemp(prefix='resid_OSL_', suffix='.nii')
        _, self.F = mkstemp(prefix='F_', suffix='.nii')
        _, self.resid = mkstemp(prefix='resid_', suffix='.nii')
        # Use a temp directory for the model.output_T images
        self.out_dir = mkdtemp()

    def tearDown(self):
        os.remove(self.ar1)
        os.remove(self.resid_OLS)
        os.remove(self.F)
        os.remove(self.resid)
        rmtree(self.out_dir)

    # FIXME: This does many things, but it does not test any values
    # with asserts.
    def testrun(self):
        funcim = load_image(funcfile)
        fmriims = fromimage(funcim, volume_start_times=2.)

        f1 = ExperimentalQuantitative("f1", lambda t:t)
        f2 = ExperimentalQuantitative("f1", lambda t:t**2)
        f3 = ExperimentalQuantitative("f1", lambda t:t**3)

        f = f1 + f2 + f3
        c = Contrast(f1, f)
        c.compute_matrix(fmriims.volume_start_times)
        c2 = Contrast(f1 + f2, f)
        c2.compute_matrix(fmriims.volume_start_times)

        outputs = []
        outputs.append(model.output_AR1(self.ar1, fmriims, clobber=True))
        outputs.append(model.output_resid(self.resid_OLS, fmriims, 
                                          clobber=True))
        ols = model.OLS(fmriims, f, outputs)
        ols.execute()

        outputs = []
        out_fn = os.path.join(self.out_dir, 'out.nii')
        outputs.append(model.output_T(out_fn, c, fmriims, clobber=True))
        outputs.append(model.output_F(self.F, c2, fmriims, clobber=True))
        outputs.append(model.output_resid(self.resid, fmriims, clobber=True))
        rho = load_image(self.ar1)
        ar = model.AR1(fmriims, f, rho, outputs)
        ar.execute()







