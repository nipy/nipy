import test_fmristat
import unittest

def suite():
    return unittest.TestSuite([test_fmristat.suite()])
