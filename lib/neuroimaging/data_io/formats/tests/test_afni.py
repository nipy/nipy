import unittest, os
import numpy as N

from numpy.testing import NumpyTestCase

from neuroimaging.data_io.formats import afni
from neuroimaging.utils.tests.data import repository
from neuroimaging.core.api import Image

from neuroimaging.utils.test_decorators import slow, data

class test_AFNI(NumpyTestCase):

    def data_setUp(self):
        # This data set is part of the AFNI test data package
        # it is anatomical with shape (124,256,256)
        self.image = afni.AFNI("anat+orig", datasource=repository)

    def setUp(self):
        pass


class test_AFNIPrint(test_AFNI):

    @data
    def test_print(self):
        print self.image

class test_AFNITransform(test_AFNI):

    @data
    def test_transform(self):
        t = self.image.grid.mapping.transform

##         this is good for epi_r1+orig
##         a = N.array([[   2.5  ,    0.   ,    0.   ,    0.   ,    0.   ],
##                      [   0.   ,    5.   ,    0.   ,    0.   ,  -69.   ],
##                      [   0.   ,    0.   ,   -3.75 ,    0.   ,  118.125],
##                      [   0.   ,    0.   ,    0.   ,   -3.75 ,  118.125],
##                      [   0.   ,    0.   ,    0.   ,    0.   ,    1.   ]])

        a = N.array([[  -1.20000005,    0.        ,    0.        ,   70.],
                     [   0.,   -0.9375    ,    0.        ,  119.53119659],
                     [   0.,    0.        ,    0.9375    , -119.53119659],
                     [   0.,    0.        ,    0.        ,    1.        ]])
        N.testing.assert_almost_equal(t,a)

    @data
    def test_shape(self):
        self.assertEquals(tuple(self.image.grid.shape), (124,256,256))

## class AFNIWriteTest(AFNITest):

##     # this fails due to formatting!
##     def test_writehdr(self):
##         new = file('tmp.hdr', 'w+')
##         self.image.write_header(new)
##         new.close()
##         new = file('tmp.hdr', 'r')
##         old = repository.open(self.image.header_file)
##         self.assertEquals(old.read(), new.read())

class test_AFNIRead(test_AFNI):

    @data
    def test_read(self):
        data = self.image[:]
        minmax = self.image.header['BRICK_STATS']
        dminmax = N.empty((2,), N.float32)
        dminmax[0::2] = N.reshape(data, (1,124*256*256)).min(axis=-1)
        dminmax[1::2] = N.reshape(data, (1,124*256*256)).max(axis=-1)
        N.testing.assert_almost_equal(minmax, dminmax)

class test_AFNIDataType(test_AFNI):

    @slow
    @data
    def test_datatypes(self):
        for sctype in afni.AFNI_dtype2bricktype.keys():
            
            _out = N.ones(self.image.grid.shape, sctype)
            out = Image(_out, grid=self.image.grid)
            out.tofile('out.HEAD', clobber=True, format=afni.AFNI)
            new = Image('out.HEAD', format=afni.AFNI)
            self.assertEquals(new._source.dtype.type, sctype)
            self.assertEquals(os.stat('out.BRIK').st_size,
                              N.product(self.image.grid.shape) *
                              _out.dtype.itemsize)
            N.testing.assert_almost_equal(new[:], _out)

        os.remove('out.HEAD')
        os.remove('out.BRIK')

    @slow
    @data
    def test_datatypes2(self):
        for sctype in afni.AFNI_dtype2bricktype.keys():
            for _sctype in afni.AFNI_dtype2bricktype.keys():
                _out = N.ones(self.image.grid.shape, sctype)
                out = Image(_out, grid=self.image.grid)
                out.tofile('out.HEAD', clobber=True,
                           dtype=_sctype, format=afni.AFNI)
                new = Image('out.HEAD', format=afni.AFNI)
                self.assertEquals(new._source.dtype.type, _sctype)
                self.assertEquals(os.stat('out.BRIK').st_size,
                                  N.product(self.image.grid.shape) *
                                  N.dtype(_sctype).itemsize)
                N.testing.assert_almost_equal(new[:], _out)

        os.remove('out.HEAD')
        os.remove('out.BRIK')



if __name__ == '__main__':
    unittest.main()
