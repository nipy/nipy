import unittest, os, scipy
import numpy as N
from neuroimaging.image.formats import analyze
from neuroimaging.data import retrieve

class AnalyzeTest(unittest.TestCase):

    def setUp(self):
        self.imgname = 'http://kff.stanford.edu/~jtaylo/BrainSTAT/rho.img'
#
#    def test_open(self):
#        analyze.ANALYZE(filename=self.imgname, datasource=retrieve)
#
#    def test_print(self):
#        self._open()
#        print self.image
#
#    def test_byteorder(self):
#        self._open()
#
#    def test_transform(self):
#        self._open()
#        t = self.image.grid.mapping.transform
#        a = N.array([[   2.,    0.,    0.,  -72.],
#                     [   0.,    2.,    0., -126.],
#                     [   0.,    0.,    2.,  -90.],
#                     [   0.,    0.,    0.,    1.]])
#        scipy.testing.assert_almost_equal(t, a)
#        
#    def test_shape(self):
#        self._open()
#        self.assertEquals(tuple(self.image.grid.shape), (91,109,91))
#
#    def test_writehdr(self):
#        self._open()
#        f = file('tmp.hdr', 'wb')
#        self.image.writeheader(f)
#        x = file(f.name).read()
#        os.remove('tmp.hdr')
#        y = file(self.image.hdrfilename()).read()
#        self.assertEquals(x, y)
#
#    def test_read(self):
#        self._open()
#        data = self.image.getslice(slice(4,7))
#        self.assertEquals(data.shape, (3,109,91))

def suite():
    suite = unittest.makeSuite(AnalyzeTest)
    return suite


if __name__ == '__main__':
    unittest.main()
