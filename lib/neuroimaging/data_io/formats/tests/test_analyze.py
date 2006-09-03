import unittest, os
import numpy as N

from neuroimaging.data_io.formats import analyze
from neuroimaging.utils.tests.data import repository
from neuroimaging.core.image import Image

class AnalyzeTest(unittest.TestCase):

    def setUp(self):
        self.image = analyze.ANALYZE("avg152T1", datasource=repository)


class AnalyzePrintTest(AnalyzeTest):
    def test_print(self):
        print self.image

class AnalyzeTransformTest(AnalyzeTest):

    def test_transform(self):
        t = self.image.grid.mapping.transform
        a = N.array([[   2.,    0.,    0.,  -72.],
                     [   0.,    2.,    0., -126.],
                     [   0.,    0.,    2.,  -90.],
                     [   0.,    0.,    0.,    1.]])
        N.testing.assert_almost_equal(t, a)

    def test_shape(self):
        self.assertEquals(tuple(self.image.grid.shape), (91,109,91))

        
class AnalyzeWriteTest(AnalyzeTest):

    def test_writehdr(self):
        new = file('tmp.hdr', 'wb')
        self.image.write_header(new)
        new.close()
        new = file('tmp.hdr', 'rb')
        old = file(repository.filename(self.image.header_filename()))
        for att in self.image.header:
            attname = att[0]
            trait = self.image.trait(attname)
            new_value = trait.handler.read(new)
            old_value = trait.handler.read(old)
            self.assertEquals(old_value, new_value)
        os.remove('tmp.hdr')
        old.seek(0); new.seek(0)
        self.assertEquals(old.read(), new.read())

class AnalyzeReadTest(AnalyzeTest):

    def test_read(self):
        data = self.image[slice(4,7)]
        self.assertEquals(data.shape, (3,109,91))

class AnalyzeDataTypeTest(AnalyzeTest):

    def test_datatypes(self):
        for sctype in analyze.datatypes.keys():
            
            _out = N.ones(self.image.grid.shape, sctype)
            out = Image(_out, grid=self.image.grid)
            out.tofile('out.hdr', clobber=True)
            new = Image('out.hdr')
            self.assertEquals(new._source.sctype, sctype)
            self.assertEquals(os.stat('out.img').st_size,
                              N.product(self.image.grid.shape) *
                              _out.dtype.itemsize)
            N.testing.assert_almost_equal(new[:], _out)

        os.remove('out.hdr')
        os.remove('out.img')

    def test_datatypes2(self):
        for sctype in analyze.datatypes.keys():
            for _sctype in analyze.datatypes.keys():
                _out = N.ones(self.image.grid.shape, sctype)
                out = Image(_out, grid=self.image.grid)
                out.tofile('out.hdr', clobber=True, sctype=_sctype)
                new = Image('out.hdr')
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
