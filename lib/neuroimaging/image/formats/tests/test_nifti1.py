import unittest, os
import numpy as N
from neuroimaging.image import Image
from neuroimaging.image.formats import nifti1
from neuroimaging.tests.data import repository

class NiftiTest(unittest.TestCase):

    def setUp(self):
        self.zimage = nifti1.NIFTI1("zstat1.nii", datasource=repository)
        self.image = Image("zstat1.nii", datasource=repository)

        self.zvalues = {'sizeof_hdr':348,
                        'data_type':'\x00'*10,
                        'db_name':'\x00'*18,
                        'extents':0,
                        'session_error':0,
                        'regular':'r',
                        'dim_info':0,
                        'dim':(3, 64, 64, 21, 1, 1, 1, 1),
                        'intent_p1':0.,
                        'intent_p2':0.,
                        'intent_p3':0.,
                        'intent_code':5,
                        'datatype':16,
                        'bitpix':32,
                        'slice_start':0,
                        'pixdim':(-1., 4., 4., 6., 1., 1., 1., 1.),
                        'vox_offset':352.,
                        'scl_slope':0.,
                        'scl_inter':0.,
                        'slice_end':0,
                        'slice_code':0,
                        'xyzt_units':10,
                        'cal_max':25500.,
                        'cal_min':3.,
                        'slice_duration':0.,
                        'toffset':0.,
                        'glmax':0,
                        'glmin':0,
                        'descrip':'FSL3.2beta' + '\x00'*70,
                        'aux_file':'\x00'*24,
                        'qform_code':1,
                        'sform_code':0,
                        'quatern_b':0.,
                        'quatern_c':1.,
                        'quatern_d':0.,
                        'qoffset_x':0.,
                        'qoffset_y':0.,
                        'qoffset_z':0.,
                        'srow_x':(0.,)*4,
                        'srow_y':(0.,)*4,
                        'srow_z':(0.,)*4,
                        'intent_name':'\x00'*16,
                        'magic':'n+1\x00'
                        }

    def test_print(self):
        print self.zimage

    def test_header1(self):
        for name, value in self.zvalues.items():
            setattr(self.zimage, name, value)

    def test_header2(self):
        for name, value in self.zvalues.items():
            self.assertEqual(getattr(self.zimage, name), value)

    def test_read1(self):
        y = self.zimage[:]
        N.testing.assert_approx_equal(y.min(), -8.71075057983)
        N.testing.assert_approx_equal(y.max(), 18.582529068)

    def test_write1(self):
        self.image.tofile('out.nii', clobber=True)
        os.remove('out.nii')

    def test_write2(self):
        self.image.tofile('out.img', clobber=True)
        os.remove('out.img')
        os.remove('out.hdr')

    def test_write3(self):
        rho = Image("rho.img", datasource=repository)
        rho.tofile('out.nii', clobber=True)
        os.remove('out.nii')


def suite():
    suite = unittest.makeSuite(AnalyzeTest)
    return suite


if __name__ == '__main__':
    unittest.main()
