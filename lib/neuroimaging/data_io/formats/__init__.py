#import format, binary, analyze, afni, ecat7, nifti1, utils

def test(level=1, verbosity=1):
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)
