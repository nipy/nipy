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
        f = file('tmp.hdr', 'wb')
        self.image.writeheader(f)
        x = file(f.name).read()
        os.remove('tmp.hdr')
        y = file(repository.filename(self.image.hdrfilename())).read()
        self.assertEquals(x, y)

    def test_read(self):
        data = self.image.getslice(slice(4,7))
        self.assertEquals(data.shape, (3,109,91))

def suite():
    suite = unittest.makeSuite(AnalyzeTest)
    return suite


if __name__ == '__main__':
    unittest.main()
