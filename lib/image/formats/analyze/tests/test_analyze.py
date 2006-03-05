import unittest, os, scipy
import numpy as N
from neuroimaging.image.formats import analyze

class AnalyzeTest(unittest.TestCase):

    def _open(self):
        imgname = '/usr/share/BrainSTAT/repository/kff.stanford.edu/~jtaylo/BrainSTAT/avg152T1.img'
        self.img = analyze.ANALYZE(filename=imgname)

    def test_print(self):
        self._open()
        print self.img

    def test_byteorder(self):
        self._open()

    def test_open(self):
        self._open()

    def test_transform(self):
        self._open()
        t = self.img.grid.warp.transform
        a = N.array([[   2.,    0.,    0.,  -72.],
                     [   0.,    2.,    0., -126.],
                     [   0.,    0.,    2.,  -90.],
                     [   0.,    0.,    0.,    1.]])
        scipy.testing.assert_almost_equal(t, a)
        
    def test_shape(self):
        self._open()
        self.assertEquals(tuple(self.img.grid.shape), (91,109,91))

    def test_writehdr(self):
        self._open()
        f = file('tmp.hdr', 'wb')
        self.img.writeheader(f)
        x = file(f.name).read()
        os.remove('tmp.hdr')
        y = file(self.img.hdrfilename()).read()
        self.assertEquals(x, y)

    def test_read(self):
        self._open()
        data = self.img.read((0,)*3, self.img.grid.shape)
        self.assertEquals(data.shape, (91,109,91))

if __name__ == '__main__':
    unittest.main()
