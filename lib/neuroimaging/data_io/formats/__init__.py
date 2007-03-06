import binary, format, utils
import afni, nifti1, analyze, ecat7

def test(level=1, verbosity=1):
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)
