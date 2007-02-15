import unittest, os
import numpy as N

from neuroimaging.utils.test_decorators import data

from neuroimaging.data_io.formats import ecat7
from neuroimaging.utils.tests.data import repository
from neuroimaging.core.api import Image
from neuroimaging.data_io.formats.ecat7 import Ecat7

class EcatTest(unittest.TestCase):

    def setUp(self):
        pass

    def data_setUp(self):
        # this is an FDG file
        # files can have 1frame or many frames
        # this file has 8 frames
        self.ecat = ecat7.Ecat7("FDG-de.v",datasource=repository)
        self.image1 = Image(self.ecat.frames[0])
        self.image8 = Image(self.ecat.frames[7])


class EcatMainHeaderTest(EcatTest):
    @data
    def test_header_print(self):
        for field,value in self.ecat.header.items():
            print '%s: %s'%(field,str(value))
            
class EcatFramesTest(EcatTest):
    @data
    def test_number_frames(self):
        header_nframes = self.ecat.header['num_frames']
        mlist_nframes = self.ecat.mlist.shape[1]
        N.testing.assert_approx_equal(header_nframes,mlist_nframes)
    @data
    def test_frame_max(self):
        y1 = self.image1[:]
        y8 = self.image8[:]
        N.testing.assert_approx_equal(y1.max(),56099.20703125)
        N.testing.assert_approx_equal(y8.max(),78144.84375)
    @data
    def test_image_shape(self):
        shape = [47, 256, 256]
        self.assertEqual(self.image1.shape,shape)

def suite():
    suite = unittest.makeSuite(EcatTest)
    return suite


if __name__ == '__main__':
    unittest.main()
