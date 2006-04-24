import unittest, os, scipy, glob, sets
import numpy as N
from neuroimaging.image import Image, interpolation
from neuroimaging.visualization import viewer, slices, arrayview
from neuroimaging.reference import slices as rslices
import pylab

class VisualizationTest(unittest.TestCase):

    def setUp(self):
        self.repo = neuroimaging.tests.data.repository
        self.img = BaseImage('rho.img', datasource=self.repo)
        self.interpolator = interpolation.ImageInterpolator(self.img)
        self.m = float(self.img.readall().min())
        self.M = float(self.img.readall().max())

    def test_array(self):
        self.image.array

if __name__ == '__main__':
    unittest.main()
