import numpy as np

from nipy.testing import *

from nipy.modalities.fmri import functions




class test_functions(TestCase):

    def test_interpolate_confound(self):
        # Test for single and multiple regressors
        T = np.arange(100)
        C = np.random.normal(size=(T.shape))
        ic = functions.InterpolatedConfound(times=T, values=C)
        assert np.allclose(C, ic(T))
        C = np.random.normal(size=(2,100))
        ic = functions.InterpolatedConfound(times=T, values=C)
        assert np.allclose(C, ic(T))




