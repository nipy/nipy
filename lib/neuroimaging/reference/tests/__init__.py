import test_axis
import test_grid
import unittest

def suite():
    return unittest.TestSuite([test_axis.suite(),
                               test_grid.suite()])

