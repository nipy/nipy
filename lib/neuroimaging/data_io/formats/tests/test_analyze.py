import unittest, os
import numpy as N

from numpy.testing import NumpyTestCase

from neuroimaging.utils.test_decorators import slow

from neuroimaging.data_io.formats import analyze
from neuroimaging.utils.tests.data import repository
from neuroimaging.core.api import Image
from neuroimaging.data_io.formats.analyze import Analyze

class test_Analyze(NumpyTestCase):

    def setUp(self):
        # this file header has dim = [4 91 109 91 1 0 0 0]
        # some tests will change if we decide to squeeze out the 4th dim
        # I have, for now, decided to squeeze the 4th dim...
        self.image = analyze.Analyze("avg152T1", datasource=repository)


class test_AnalyzePrintTest(test_Analyze):
    def test_print(self):
        print self.image

class test_AnalyzeTransform(test_Analyze):

    def test_transform(self):
        t = self.image.grid.mapping.transform
        """
        a = N.array([[   1.,    0.,    0.,    0.,    1.],
                     [   0.,    2.,    0.,    0.,  -72.],
                     [   0.,    0.,    2.,    0., -126.],
                     [   0.,    0.,    0.,    2.,  -90.],
                     [   0.,    0.,    0.,    0.,    1.]])

                     """
        a = N.array([[   2.,    0.,    0.,  -72.],
                     [   0.,    2.,    0., -126.],
                     [   0.,    0.,    2.,  -90.],
                     [   0.,    0.,    0.,    1.]])

        N.testing.assert_almost_equal(t, a)

    def test_shape(self):
        self.assertEquals(tuple(self.image.grid.shape), (91,109,91))

        
class test_AnalyzeWrite(test_Analyze):

    def test_writehdr(self):
        new = file('tmp.hdr', 'wb')
        self.image.write_header(new)
        new.close()
        new = file('tmp.hdr', 'rb')
        old = repository.open(self.image.header_file)
        self.assertEquals(old.read(), new.read())

class test_AnalyzeRead(test_Analyze):

    def test_read(self):
        data = self.image[:,4:7]
        self.assertEquals(data.shape, (91,3,91))

class test_AnalyzeDataType(test_Analyze):

    @slow
    def test_datatypes(self):
        for sctype in analyze.sctype2datatype.keys():
            
            _out = N.ones(self.image.grid.shape, sctype)
            out = Image(_out, grid=self.image.grid)
            out.tofile('out.hdr', clobber=True, format=Analyze)
            new = Image('out.hdr', format=Analyze)
            self.assertEquals(new._source.dtype.type, sctype)
            self.assertEquals(os.stat('out.img').st_size,
                              N.product(self.image.grid.shape) *
                              _out.dtype.itemsize)
            N.testing.assert_almost_equal(new[:], _out)

        os.remove('out.hdr')
        os.remove('out.img')

    @slow
    def test_datatypes2(self):
        for sctype in analyze.sctype2datatype.keys():
            for _sctype in analyze.sctype2datatype.keys():
                _out = N.ones(self.image.grid.shape, sctype)
                out = Image(_out, grid=self.image.grid)
                out.tofile('out.hdr', clobber=True, dtype=_sctype, format=Analyze)
                new = Image('out.hdr', format=Analyze)
                self.assertEquals(new._source.dtype.type, _sctype)
                self.assertEquals(os.stat('out.img').st_size,
                                  N.product(self.image.grid.shape) *
                                  N.dtype(_sctype).itemsize)
                N.testing.assert_almost_equal(new[:], _out)

        os.remove('out.hdr')
        os.remove('out.img')

def suite():
    suite = unittest.makeSuite(test_Analyze)
    return suite


if __name__ == '__main__':
    unittest.main()

