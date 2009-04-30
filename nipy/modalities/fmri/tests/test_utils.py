import numpy as np
from nipy.testing import *

from nipy.modalities.fmri.utils import CutPoly, WaveFunction, ConvolveFunctions

class test_util(TestCase):
    

    def test_ConvolveFunctions(self):
        """
        The integral of the normalised convolution should be equal
        to the product of the integrals of the original functions.
        """
        dt = 0.01
        t = np.arange(0, 10.0, dt)
        f1 = WaveFunction(2, 1, 1)
        f2 = WaveFunction(4, 1, 1)
        fa = ConvolveFunctions(f1, f2, [t[0], t[-1]], dt, normalize=[1,1])
        int_f1 = dt*f1(t).sum()
        int_f2 = dt*f2(t).sum()
        int_fa = dt*fa(t).sum()
        assert_approx_equal(int_f1*int_f2, int_fa)








