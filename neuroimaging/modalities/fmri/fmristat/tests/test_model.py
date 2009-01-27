from neuroimaging.testing import *

import neuroimaging.modalities.fmri.fmristat.model as model
from neuroimaging.testing import funcfile
from neuroimaging.modalities.fmri.api import fromimage
from neuroimaging.core.api import load_image

from neuroimaging.modalities.fmri.protocol import Formula, ExperimentalQuantitative
from neuroimaging.fixes.scipy.stats.models.contrast import Contrast

# Load in the data

class test_fMRIstat_model(TestCase):

    def setUp(self):
        pass

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
        outputs.append(model.output_AR1("ar1.nii", fmriims, clobber=True))
        outputs.append(model.output_resid("resid_OLS.nii", fmriims, clobber=True))
        ols = model.OLS(fmriims, f, outputs)
        ols.execute()

        outputs = []
        outputs.append(model.output_T("out_%(stat)s.nii", c, fmriims, clobber=True))
        outputs.append(model.output_F("F.nii", c2, fmriims, clobber=True))
        outputs.append(model.output_resid("resid.nii", fmriims, clobber=True))
        rho = load_image("ar1.nii")
        ar = model.AR1(fmriims, f, rho, outputs)
        ar.execute()

        os.remove('ar1.nii')
        os.remove('F.nii')
        os.remove('resid.nii')
        os.remove('resid_OLS.nii')
        os.remove('out_t.nii')
        os.remove('out_sd.nii')
        os.remove('out_effect.nii')







