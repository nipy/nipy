import unittest

from neuroimaging.ui.visualization.tests import \
  test_visualization, test_montage

def suite():
    return unittest.TestSuite([test_visualization.suite(),
                               test_multiplot.suite(),
                               test_montage.suite()])
