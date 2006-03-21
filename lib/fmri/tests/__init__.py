import test_fmri
import test_protocol
import unittest

import neuroimaging.fmri.fmristat.tests as fmristat_tests

def suite():
    return unittest.TestSuite([fmristat_tests.suite()])
