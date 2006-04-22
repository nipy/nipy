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
        self.image = baseimage.BaseImage(str(imgpath))
        #self.interpolator = interpolation.ImageInterpolator(self.image)
        #self.m = float(self.image.readall().min())
        #self.M = float(self.image.readall().max())

    def test_array(self):
        self.image.array

    def test_arrayview(self):
        arrayview(self.image.array)

if __name__ == '__main__':
    unittest.main()
