import numpy as N

from numpy.testing import NumpyTest, NumpyTestCase

from neuroimaging.modalities.fmri import functions

from neuroimaging.utils.testutils import make_doctest_suite
test_suite = make_doctest_suite('neuroimaging.modalities.fmri.functions')

class test_functions(NumpyTestCase):

    def test_interpolate_confound(self):
        # Test for single and multiple regressors
        T = N.arange(100)
        C = N.random.normal(size=(T.shape))
        ic = functions.InterpolatedConfound(times=T, values=C)
        assert N.allclose(C, ic(T))
        C = N.random.normal(size=(2,100))
        ic = functions.InterpolatedConfound(times=T, values=C)
        assert N.allclose(C, ic(T))


if __name__ == '__main__':
    NumpyTest.run()
