import numpy as N
from numpy.testing import NumpyTest, NumpyTestCase

from neuroimaging.core.image.base_image import BaseImage, ArrayImage
from neuroimaging.core.reference.grid import SamplingGrid


class test_BaseImage(NumpyTestCase):


    def test_init(self):
        data = [[[1,2,3], [4,5,6], [7,8,9]],
                [[1,2,3], [4,5,6], [7,8,9]],
                [[1,2,3], [4,5,6], [7,8,9]]]
        grid = SamplingGrid.identity((3,3,3))
        img = BaseImage(data, grid, N.int32)

    def test_get(self):
        data = [[[1,2,3], [4,5,6], [7,8,9]],
                [[1,2,3], [4,5,6], [7,8,9]],
                [[1,2,3], [4,5,6], [7,8,9]]]
        grid = SamplingGrid.identity((3,3,3))
        img = BaseImage(data, grid, N.int32)

        x = img[0]
        self.assertEquals(x, [[1,2,3],[4,5,6],[7,8,9]])

    def test_set(self):
        data = [[[1,2,3], [4,5,6], [7,8,9]],
                [[1,2,3], [4,5,6], [7,8,9]],
                [[1,2,3], [4,5,6], [7,8,9]]]
        grid = SamplingGrid.identity((3,3,3))
        img = BaseImage(data, grid, N.int32)

        row = [[9,8,7],[6,5,4],[3,2,1]]
        img[0] = row
        self.assertEquals(img[0], row)


class test_ArrayImage(NumpyTestCase):


    def test_init(self):
        data = N.array([[[1,2,3], [4,5,6], [7,8,9]],
                        [[1,2,3], [4,5,6], [7,8,9]],
                        [[1,2,3], [4,5,6], [7,8,9]]])
        img = ArrayImage(data)

    def test_get(self):
        data = N.array([[[1,2,3], [4,5,6], [7,8,9]],
                        [[1,2,3], [4,5,6], [7,8,9]],
                        [[1,2,3], [4,5,6], [7,8,9]]])
        img = ArrayImage(data)

        x = img[0]
        N.array_equal(x, [[1,2,3],[4,5,6],[7,8,9]])

    def test_set(self):
        data = [[[1,2,3], [4,5,6], [7,8,9]],
                [[1,2,3], [4,5,6], [7,8,9]],
                [[1,2,3], [4,5,6], [7,8,9]]]
        grid = SamplingGrid.identity((3,3,3))
        img = BaseImage(data, grid, N.int32)

        row = [[9,8,7],[6,5,4],[3,2,1]]
        img[0] = row
        N.array_equal(img[0], row)
        #self.assertEquals(img[0], row)



if __name__ == '__main__':
    NumpyTest.run()
