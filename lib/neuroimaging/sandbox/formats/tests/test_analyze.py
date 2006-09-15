import unittest, os
import numpy as N

from neuroimaging.sandbox.formats import analyze
from neuroimaging.utils.tests.data import repository
from neuroimaging.core.image.image import Image
from neuroimaging.sandbox.formats.analyze import Analyze

class AnalyzeTest(unittest.TestCase):

    def setUp(self):
        # this file header has dim = [4 91 109 91 1 0 0 0]
        # some tests will change if we decide to squeeze out the 4th dim
        self.image = analyze.Analyze("avg152T1", datasource=repository)


class AnalyzePrintTest(AnalyzeTest):
    def test_print(self):
        print self.image

class AnalyzeTransformTest(AnalyzeTest):

    def test_transform(self):
        t = self.image.grid.mapping.transform
        a = N.array([[   1.,    0.,    0.,    0.,    1.],
                     [   0.,    2.,    0.,    0.,  -72.],
                     [   0.,    0.,    2.,    0., -126.],
                     [   0.,    0.,    0.,    2.,  -90.],
                     [   0.,    0.,    0.,    0.,    1.]])

        N.testing.assert_almost_equal(t, a)

    def test_shape(self):
        self.assertEquals(tuple(self.image.grid.shape), (1,91,109,91))

        
class AnalyzeWriteTest(AnalyzeTest):

    def test_writehdr(self):
        new = file('tmp.hdr', 'wb')
        self.image.write_header(new)
        new.close()
        new = file('tmp.hdr', 'rb')
        old = file(repository.filename(self.image.header_file))
        self.assertEquals(old.read(), new.read())

class AnalyzeReadTest(AnalyzeTest):

    def test_read(self):
        data = self.image[:,4:7]
        self.assertEquals(data.shape, (1,3,109,91))

class AnalyzeDataTypeTest(AnalyzeTest):

    def test_datatypes(self):
        for sctype in analyze.sctype2datatype.keys():
            
            _out = N.ones(self.image.grid.shape, sctype)
            out = Image(_out, grid=self.image.grid)
            out.tofile('out.hdr', clobber=True, format=Analyze)
            new = Image('out.hdr', format=Analyze)
            self.assertEquals(new._source.sctype, sctype)
            self.assertEquals(os.stat('out.img').st_size,
                              N.product(self.image.grid.shape) *
                              _out.dtype.itemsize)
            N.testing.assert_almost_equal(new[:], _out)

        os.remove('out.hdr')
        os.remove('out.img')

    def test_datatypes2(self):
        for sctype in analyze.sctype2datatype.keys():
            for _sctype in analyze.sctype2datatype.keys():
                _out = N.ones(self.image.grid.shape, sctype)
                out = Image(_out, grid=self.image.grid)
                out.tofile('out.hdr', clobber=True, sctype=_sctype, format=Analyze)
                new = Image('out.hdr', format=Analyze)
                self.assertEquals(new._source.sctype, _sctype)
                self.assertEquals(os.stat('out.img').st_size,
                                  N.product(self.image.grid.shape) *
                                  N.dtype(_sctype).itemsize)
                N.testing.assert_almost_equal(new[:], _out)

        os.remove('out.hdr')
        os.remove('out.img')

def suite():
    suite = unittest.makeSuite(AnalyzeTest)
    return suite


if __name__ == '__main__':
    unittest.main()
