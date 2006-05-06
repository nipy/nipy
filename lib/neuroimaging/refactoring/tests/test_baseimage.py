import unittest
from neuroimaging.refactoring import baseimage
from neuroimaging.visualization.arrayview import arrayview

class BaseImageTest(unittest.TestCase):

    def setUp(self):
        imgpath = neuroimaging.tests.data.datapath.joinpath("rho.img")
        self.image = baseimage.image_factory(str(imgpath))

    def test_array(self):
        self.image.raw_array

    def test_arrayview(self):
        arrayview(self.image.raw_array)


if __name__ == '__main__': unittest.main()
