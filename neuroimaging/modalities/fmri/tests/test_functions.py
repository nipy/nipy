import numpy as N

from neuroimaging.testing import *

from neuroimaging.modalities.fmri import functions




class test_functions(TestCase):

    def test_interpolate_confound(self):
        # Test for single and multiple regressors
        T = N.arange(100)
        C = N.random.normal(size=(T.shape))
        ic = functions.InterpolatedConfound(times=T, values=C)
        assert N.allclose(C, ic(T))
        C = N.random.normal(size=(2,100))
        ic = functions.InterpolatedConfound(times=T, values=C)
        assert N.allclose(C, ic(T))




