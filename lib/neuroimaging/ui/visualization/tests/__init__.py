import unittest

from neuroimaging.ui.visualization.tests import \
  test_visualization, test_montage, test_multiplot

def suite():
    return unittest.TestSuite([test_visualization.suite(),
                               test_montage.suite(),
                               test_multiplot.suite()])
