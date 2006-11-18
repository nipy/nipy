import test_afni
import unittest

def suite():
    return unittest.TestSuite((test_afni.suite(),))
