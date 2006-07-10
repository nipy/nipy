import unittest, os
import numpy as N
from neuroimaging.image.formats import analyze
from neuroimaging.tests.data import repository

class AnalyzeTest(unittest.TestCase):

    def setUp(self):
        self.image = analyze.ANALYZE("avg152T1", datasource=repository)

    def test_print(self):
        print self.image

    def test_transform(self):
        t = self.image.grid.mapping.transform
        a = N.array([[   2.,    0.,    0.,  -72.],
                     [   0.,    2.,    0., -126.],
                     [   0.,    0.,    2.,  -90.],
                     [   0.,    0.,    0.,    1.]])
        N.testing.assert_almost_equal(t, a)
        
    def test_shape(self):
        self.assertEquals(tuple(self.image.grid.shape), (91,109,91))

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

    def test_read(self):
        data = self.image.getslice(slice(4,7))
        self.assertEquals(data.shape, (3,109,91))

def suite():
    suite = unittest.makeSuite(AnalyzeTest)
    return suite


if __name__ == '__main__':
    unittest.main()
