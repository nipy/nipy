import unittest
from neuroimaging.refactoring import baseimage
from neuroimaging.tests.data import repository
from neuroimaging.visualization.arrayview import arrayview

class BaseImageTest(unittest.TestCase):

    def setUp(self):
        self.image = baseimage.image("rho.img", datasource=repository)

    def test_array(self):
        self.image.array

    def test_arrayview(self):
        arrayview(self.image.array)


if __name__ == '__main__': unittest.main()
