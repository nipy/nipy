import test_analyze
import test_nifti1
import unittest

def suite():
    return unittest.TestSuite(test_analyze.suite())
    

