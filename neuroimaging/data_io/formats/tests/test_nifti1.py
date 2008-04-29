import os, copy
from cStringIO import StringIO

import numpy as N
import numpy.random as R

from numpy.testing import NumpyTest, NumpyTestCase

from neuroimaging.utils.test_decorators import slow

from neuroimaging.core.api import Image, load_image, save_image
from neuroimaging.testing import anatfile, funcfile
from neuroimaging.data_io.formats import nifti1
from neuroimaging.utils.odict import odict

class test_Nifti(NumpyTestCase):

    def setUp(self):
        self.anat = load_image(anatfile)
        self.func = load_image(funcfile)

class test_NiftiPrint(test_Nifti):

    def test_print(self):
        print >> StringIO(), self.zimage

class test_NiftiHeader(test_Nifti):

    def test_header1(self):       
        zvalues = odict((
            ('sizeof_hdr',348),
            ('data_type','\x00'*10),
            ('db_name','\x00'*18),
            ('extents',0),
            ('session_error',0),
            ('regular','r'),
            ('dim_info',0),
            ('dim',[3, 64, 64, 21, 1, 1, 1, 1]),
            ('intent_p1',0.),
            ('intent_p2',0.),
            ('intent_p3',0.),
            ('intent_code',5),
            ('datatype',16),
            ('bitpix',32),
            ('slice_start',0),
            ('pixdim',[-1., 4., 4., 6., 1., 1., 1., 1.]),
            ('vox_offset',352.),
            ('scl_slope',0.),
            ('scl_inter',0.),
            ('slice_end',0),
            ('slice_code',0),
            ('xyzt_units',10),
            ('cal_max',25500.),
            ('cal_min',3.),
            ('slice_duration',0.),
            ('toffset',0.),
            ('glmax',0),
            ('glmin',0),
            ('descrip','FSL3.2beta' + '\x00'*70),
            ('aux_file','\x00'*24),
            ('qform_code',1),
            ('sform_code',0),
            ('quatern_b',0.),
            ('quatern_c',1.),
            ('quatern_d',0.),
            ('qoffset_x',0.),
            ('qoffset_y',0.),
            ('qoffset_z',0.),
            ('srow_x',[0.]*4),
            ('srow_y',[0.]*4),
            ('srow_z',[0.]*4),
            ('intent_name','\x00'*16),
            ('magic','n+1\x00'),
        ))

        for name, value in zvalues.items():
            self.zimage.header[name] = value


class test_NiftiRead(test_Nifti):

    def test_read1(self):
        y = N.asarray(self.anatfile)
        N.testing.assert_approx_equal(y.min(), -8.71075057983)
        N.testing.assert_approx_equal(y.max(), 18.582529068)

class test_NiftiWrite(test_Nifti):

    def test_write1(self):
        save_file(self.anatfile, 'out.nii', clobber=True, dtype=N.float64)
        out = load_image("out.nii")
        self.assertEquals(out._source.dtype.type, N.float64)
        N.testing.assert_almost_equal(out.grid.mapping.transform,
                                      self.anatfile.grid.mapping.transform)
        os.remove('out.nii')

    def test_write2(self):
        self.image.tofile('out.hdr', clobber=True)
        # these would fail, because now creating a Nifti file
        # always creats a single .nii file
        #os.remove('out.img')
        #os.remove('out.hdr')
        #os.remove('out.nii')


## Comment out the 'slow' test for now...
"""
class test_NiftiDataType(test_Nifti):

    @slow
    def test_datatypes(self):
        for sctype in nifti1.sctype2datatype.keys():
            _out = N.ones(self.anatfile.grid.shape, sctype)
            out = Image(_out, self.anatfile.grid)
            save_image(out, 'out.nii', clobber=True)
            new = load_image('out.nii')
            self.assertEquals(new.fmt.header['datatype'],
                              nifti1.sctype2datatype[sctype])
            self.assertEquals(new.fmt.dtype.type, sctype)
            self.assertEquals(new.fmt.header['vox_offset'], 352)
            self.assertEquals(os.stat('out.nii').st_size,
                              N.product(self.image.grid.shape) *
                              _out.dtype.itemsize +
                              new.fmt.header['vox_offset'])
            N.testing.assert_almost_equal(N.asarray(new), _out)
        os.remove('out.nii')

    @slow
    def test_datatypes2(self):
        for sctype in nifti1.sctype2datatype.keys():
            for _sctype in nifti1.sctype2datatype.keys():
                _out = N.ones(self.anatfile.grid.shape, sctype)
                out = Image(_out, self.anatfile.grid)
                save_image(out, 'out.nii', clobber=True)
                new = load_image('out.nii')
                self.assertEquals(new.fmt.header['datatype'],
                                  nifti1.sctype2datatype[_sctype])
                self.assertEquals(new.fmt.dtype.type, _sctype)
                self.assertEquals(new.fmt.header['vox_offset'], 352.0)
                self.assertEquals(os.stat('out.nii').st_size,
                                  N.product(self.image.grid.shape) *
                                  N.dtype(_sctype).itemsize +
                                  new.fmt.header['vox_offset'])
                N.testing.assert_almost_equal(N.asarray(new), _out)


        os.remove('out.nii')
"""

if __name__ == '__main__':
    NumpyTest().run()
