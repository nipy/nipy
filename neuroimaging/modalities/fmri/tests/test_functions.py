import numpy as N

from numpy.testing import NumpyTest, NumpyTestCase

from neuroimaging.modalities.fmri import functions

from neuroimaging.utils.testutils import make_doctest_suite
test_suite = make_doctest_suite('neuroimaging.modalities.fmri.functions')

class test_functions(NumpyTestCase):

    def test_interpolate_confound(self):
        # Test for multiple columns of regressors
        T = N.arange(100)
        C = N.random.normal(size=(2,100))
        ic = functions.InterpolatedConfound(T, C)


if __name__ == '__main__':
    NumpyTest.run()
