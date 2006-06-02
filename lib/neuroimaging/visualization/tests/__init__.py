import test_visualization
import test_montage
import unittest

def suite():
    return unittest.TestSuite([test_visualization.suite(),
                               test_montage.suite()])
