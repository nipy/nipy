import unittest, os, copy

import numpy as N
import numpy.random as R

from neuroimaging.image import Image
from neuroimaging.image.formats import nifti1
from neuroimaging.image.formats.binary import BinaryFormatError
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

class NiftiPrintTest(NiftiTest):

    def test_print(self):
        print self.zimage

class NiftiHeaderTest(NiftiTest):

    def test_header1(self):
        for name, value in self.zvalues.items():
            setattr(self.zimage, name, value)

    def test_header2(self):
        for name, value in self.zvalues.items():
            self.assertEqual(getattr(self.zimage, name), value)

class NiftiReadTest(NiftiTest):

    def test_read1(self):
        y = self.zimage[:]
        N.testing.assert_approx_equal(y.min(), -8.71075057983)
        N.testing.assert_approx_equal(y.max(), 18.582529068)

class NiftiWriteTest(NiftiTest):

    def test_write1(self):
        self.image.tofile('out.nii', clobber=True, sctype=N.float64)
        out = Image('out.nii')
        self.assertEquals(out.image.sctype, N.float64)
        os.remove('out.nii')

    def test_write2(self):
        self.image.tofile('out.img', clobber=True)
        os.remove('out.img')
        os.remove('out.hdr')

    def test_write4(self):
        self.image.tofile('out.img', clobber=True, sctype=N.float64)
        new = nifti1.NIFTI1('out.hdr')
        self.assertEquals(out.image.sctype, N.float64)
        os.remove('out.img')
        os.remove('out.hdr')

    def test_write3(self):
        rho = Image("rho.img", datasource=repository)
        rho.tofile('out.nii', clobber=True)
        os.remove('out.nii')

class NiftiModifyHeaderTest(NiftiTest):

    def test_add_header_attribute1(self):
        try:
            self.zimage.add_header_attribute('test', 'f', 0.0)
        except BinaryFormatError:
            pass

    def test_add_header_attribute2(self):
        newheader = copy.copy(list(self.zimage.header))
        x = R.standard_normal((30,))
        newheader.append(('x', '30d', tuple(x)))
        testi = nifti1.NIFTI1('out2.nii', mode='w',
                              header=newheader, grid=self.zimage.grid,
                              clobber=True)
        testi.write_header()
        del(testi)
        test2 = nifti1.NIFTI1('out2.nii', header=newheader)
        N.testing.assert_almost_equal(test2.x, x)


    def test_add_header_attribute3(self):
        newheader = copy.copy(list(self.zimage.header))
        newheader.append(('x', '30d', (0.0,)*30))
        testi = nifti1.NIFTI1('out3.hdr', mode='w',
                             header=newheader, grid=self.zimage.grid, clobber=True)
        y = R.standard_normal((3,))
        testi.add_header_attribute('y', '3d', tuple(y))
        testi.y = y
        testi.remove_header_attribute('x')
        testi.write_header()
        self.assertEquals(testi.header_length, os.stat('out3.hdr').st_size)
        header = testi.header

        del(testi)
        test2 = nifti1.NIFTI1('out3.hdr', header=header)
        self.assertEquals(test2.header_length, os.stat('out3.hdr').st_size)
        self.assertEquals(hasattr(test2, 'x'), False)
        self.assertEquals(test2.y, tuple(y))
        os.remove('out3.hdr')
        os.remove('out3.img')

class NiftiDataTypeTest(NiftiTest):

    def test_datatypes(self):
        for sctype in nifti1.datatypes.keys():
            _out = N.ones(self.zimage.grid.shape, sctype)
            out = Image(_out, grid=self.zimage.grid)
            out.tofile('out.nii', clobber=True)
            new = Image('out.nii')
            self.assertEquals(new.source.datatype, nifti1.datatypes[sctype])
            self.assertEquals(new.source.sctype, sctype)
            self.assertEquals(new.source.vox_offset, 352)
            self.assertEquals(os.stat('out.nii').st_size,
                              N.product(self.image.grid.shape) *
                              _out.dtype.itemsize + new.source.vox_offset)
            N.testing.assert_almost_equal(new[:], _out)
        os.remove('out.nii')

    def test_datatypes2(self):
        for sctype in nifti1.datatypes.keys():
            for _sctype in nifti1.datatypes.keys():
                _out = N.ones(self.zimage.grid.shape, sctype)
                out = Image(_out, grid=self.zimage.grid)
                out.tofile('out.nii', clobber=True, sctype=_sctype)
                new = Image('out.nii')
                self.assertEquals(new.source.datatype, nifti1.datatypes[_sctype])
                self.assertEquals(new.source.sctype, _sctype)
                self.assertEquals(new.source.vox_offset, 352)
                self.assertEquals(os.stat('out.nii').st_size,
                                  N.product(self.image.grid.shape) *
                                  N.dtype(_sctype).itemsize +
                                  new.source.vox_offset)
                N.testing.assert_almost_equal(new[:], _out)


        os.remove('out.nii')



        
def suite():
    suite = unittest.makeSuite(AnalyzeTest)
    return suite


if __name__ == '__main__':
    unittest.main()
