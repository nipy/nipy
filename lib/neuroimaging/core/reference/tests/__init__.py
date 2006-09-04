import unittest

from neuroimaging.core.reference.tests import test_axis, test_grid

def suite():
    return unittest.TestSuite([test_axis.suite(),
                               test_grid.suite()])

