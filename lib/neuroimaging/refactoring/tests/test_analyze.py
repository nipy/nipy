import unittest
from neuroimaging.refactoring.analyze import AnalyzeImage
from neuroimaging.tests.data import repository
from neuroimaging.visualization.arrayview import arrayview

class AnalyzeImageTest(unittest.TestCase):

    def setUp(self):
        self.image = AnalyzeImage("rho", datasource=repository)

    def test_header(self):
        self.image.raw_array

    def test_arrayview(self):
        arrayview(self.image.raw_array)


if __name__ == '__main__': unittest.main()
