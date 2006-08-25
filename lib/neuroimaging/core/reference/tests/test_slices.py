import unittest
import numpy as N

from neuroimaging.reference.slices import bounding_box
from neuroimaging.reference.grid import SamplingGrid

class SliceTest(unittest.TestCase):

    def test_bounding_box(self):
        shape = (10, 10, 10)
        grid = SamplingGrid.identity(shape)
        print bounding_box(grid)

if __name__ == '__main__':
    unittest.main()
