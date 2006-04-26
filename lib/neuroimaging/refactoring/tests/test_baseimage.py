import unittest, os, scipy, glob, sets
import numpy as N
from path import path
from neuroimaging.image import Image, interpolation
from neuroimaging.image.formats.analyze import ANALYZE
from neuroimaging.tests.data import repository
from neuroimaging.visualization import baseimage
from neuroimaging.visualization.arrayview import arrayview
from neuroimaging.reference import slices as rslices
import neuroimaging.tests.data
import pylab

class BaseImageTest(unittest.TestCase):

    def setUp(self):
        imgpath = neuroimaging.tests.data.datapath.joinpath("rho.img")
        self.image = baseimage.image_factory(str(imgpath))

    def test_array(self):
        self.image.raw_array

    def test_arrayview(self):
        arrayview(self.image.raw_array)

if __name__ == '__main__':
    unittest.main()
