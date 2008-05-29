from tempfile import NamedTemporaryFile
from StringIO import StringIO

import numpy as np

# Use nose for testing
from neuroimaging.externals.scipy.testing import *
from neuroimaging.utils.test_decorators import slow

from neuroimaging.core.api import Image, load_image, save_image
from neuroimaging.testing import anatfile
from neuroimaging.data_io.formats import nifti1

def nifti_tempfile():
    """Create and return a temporary file with nifti suffix."""
    return NamedTemporaryFile(prefix='nifti', suffix='.nii')

def default_value(fieldname):
    """Return default value from _field_defaults."""
    try:
        val = nifti1._field_defaults[fieldname]
        return val
    except KeyError:
        msg = "Nifti1 does not have default value for key '%s'" % fieldname
        raise KeyError, msg

class TestHeaderDefaults(TestCase):
    """Test default header values for nifti io."""
    def setUp(self):
        """Create a default header."""
        self.header = nifti1.create_default_header()

    def test_sizeof_hdr(self):
        key = 'sizeof_hdr'
        self.assertEqual(self.header[key], default_value(key))

    def test_scl_slope(self):
        key = 'scl_slope'
        self.assertEqual(self.header[key], default_value(key))

    def test_magic(self):
        key = 'magic'
        self.assertEqual(self.header[key], default_value(key))

    def test_pixdim(self):
        key = 'pixdim'
        self.assertEqual(self.header[key], default_value(key))

    def test_vox_offset(self):
        key = 'vox_offset'
        self.assertEqual(self.header[key], default_value(key))

class TestHeaderLittleEndian(TestCase):
    """Test read/write of little endian header for nifti io."""
    def setUp(self):
        """Create a default header and pack into a StringIO object."""
        self.formats = nifti1.struct_formats.copy()
        self.header = nifti1.create_default_header()
        self.byteorder = nifti1.utils.LITTLE_ENDIAN
        self.packed = nifti1.utils.struct_pack(self.byteorder,
                                               self.formats.values(),
                                               self.header.values())
        self.fp = StringIO()
        self.fp.write(self.packed)
        self.fp.seek(0)
        self.unpacked = nifti1.utils.struct_unpack(self.fp,
                                                   self.byteorder,
                                                   self.formats.values())

    def tearDown(self):
        self.fp.close()
        del self.fp

    def test_roundtrip(self):
        # DEV NOTE:  stopped implementing to work on numpy 1.1.0
        # Decide if this is a necessary test before going further!!!
        raise NotImplementedError, 'ADDING LITTLE ENDIAN TESTS!'

def test_scale_factor_only():
    # scale data from [0,255] to [0.0, 1.0]
    data_uint8 = np.array([0, 128, 255], dtype=np.uint8)
    scl_factor = 1.0/255    # normalize data to range [0,1]
    data_float = nifti1.scale_data(data_uint8, scale_factor=scl_factor)
    assert np.allclose(data_float.min(), 0.0, rtol=0.01)
    assert np.allclose(data_float.max(), 1.0, rtol=0.01)
    assert np.allclose(data_float.mean(), 0.5, rtol=0.01)
        
def test_scale_factor_and_intercept():
    # scale data from [-32768, 32767] to [0.0, 1.0]
    data_int16 = np.array([-32768, 0, 32767], dtype=np.int16)
    scl_factor = 1.0/(2**16)
    scl_inter = 0.5
    data_float = nifti1.scale_data(data_int16, scale_factor=scl_factor,
                                   scale_inter=scl_inter)
    # values won't be exact, but should be within desired range
    assert np.allclose(data_float.min(), 0.000, rtol=0.01)
    assert np.allclose(data_float.max(), 0.999, rtol=0.01)
    assert np.allclose(data_float.mean(), 0.5, rtol=0.1)

def test_invalid_scale_params():
    data_uint8 = np.array([0, 128, 255], dtype=np.uint8)
    # foo and bar are not kwargs searched in the scale_data function
    # they should not affect the data
    data_new = nifti1.scale_data(data_uint8, foo=5.0, bar=100.0)
    assert np.allclose(data_uint8, data_new)


"""
Some header tests to add:
test reading of byte order
valid pixdims and qfac
sensical qform and sforms
"""
def test_should_fail():
    raise NotImplementedError, 'Need to add more tests for headers.'


#
# NOTE:  Should rewrite these tests so we don't depend on a specific
#     data file.  Instead generate some image data and test on that. 
#
class test_Nifti(TestCase):
    def setUp(self):
        self.anat = load_image(anatfile)

class test_NiftiRead(test_Nifti):
    def test_read1(self):
        imgarr = np.asarray(self.anat)
        assert_approx_equal(imgarr.min(), 1910.0)
        assert_approx_equal(imgarr.max(), 7902.0)

class test_NiftiWrite(test_Nifti):
    def setUp(self):
        print self.__class__
        super(self.__class__, self).setUp()
        self.tmpfile = nifti_tempfile()

    def teardown(self):
        self.tmpfile.unlink

    def test_roundtrip_affine(self):
        save_image(self.anat, self.tmpfile.name, clobber=True, dtype=np.float64)
        outimg = load_image(self.tmpfile.name)
        assert_almost_equal(outimg.affine, self.anat.affine)

    def test_roundtrip_dtype(self):
        # Test some dtypes, presume we don't need to test all as numpy
        # should catch major dtype errors?
        # uint8
        save_image(self.anat, self.tmpfile.name, clobber=True, dtype=np.uint8)
        outimg = load_image(self.tmpfile.name)
        self.assertEquals(outimg._data.dtype.type, np.uint8,
                          'Error roundtripping dtype uint8')
        # int16
        save_image(self.anat, self.tmpfile.name, clobber=True, dtype=np.int16)
        outimg = load_image(self.tmpfile.name)
        self.assertEquals(outimg._data.dtype.type, np.int16,
                          'Error roundtripping dtype int16')
        # int32
        save_image(self.anat, self.tmpfile.name, clobber=True, dtype=np.int32)
        outimg = load_image(self.tmpfile.name)
        self.assertEquals(outimg._data.dtype.type, np.int32,
                          'Error roundtripping dtype int32')
        # float32
        save_image(self.anat, self.tmpfile.name, clobber=True, dtype=np.float32)
        outimg = load_image(self.tmpfile.name)
        self.assertEquals(outimg._data.dtype.type, np.float32,
                          'Error roundtripping dtype float32')
        # float64
        save_image(self.anat, self.tmpfile.name, clobber=True, dtype=np.float64)
        outimg = load_image(self.tmpfile.name)
        self.assertEquals(outimg._data.dtype.type, np.float64,
                          'Error roundtripping dtype float64')
        
## Comment out old tests for now...
"""
class test_NiftiDataType(test_Nifti):

    @slow
    def test_datatypes(self):
        for sctype in nifti1.sctype2datatype.keys():
            _out = np.ones(self.anatfile.grid.shape, sctype)
            out = Image(_out, self.anatfile.grid)
            save_image(out, 'out.nii', clobber=True)
            new = load_image('out.nii')
            self.assertEquals(new.fmt.header['datatype'],
                              nifti1.sctype2datatype[sctype])
            self.assertEquals(new.fmt.dtype.type, sctype)
            self.assertEquals(new.fmt.header['vox_offset'], 352)
            self.assertEquals(os.stat('out.nii').st_size,
                              np.product(self.image.grid.shape) *
                              _out.dtype.itemsize +
                              new.fmt.header['vox_offset'])
            np.testing.assert_almost_equal(np.asarray(new), _out)
        os.remove('out.nii')

    @slow
    def test_datatypes2(self):
        for sctype in nifti1.sctype2datatype.keys():
            for _sctype in nifti1.sctype2datatype.keys():
                _out = np.ones(self.anatfile.grid.shape, sctype)
                out = Image(_out, self.anatfile.grid)
                save_image(out, 'out.nii', clobber=True)
                new = load_image('out.nii')
                self.assertEquals(new.fmt.header['datatype'],
                                  nifti1.sctype2datatype[_sctype])
                self.assertEquals(new.fmt.dtype.type, _sctype)
                self.assertEquals(new.fmt.header['vox_offset'], 352.0)
                self.assertEquals(os.stat('out.nii').st_size,
                                  np.product(self.image.grid.shape) *
                                  np.dtype(_sctype).itemsize +
                                  new.fmt.header['vox_offset'])
                np.testing.assert_almost_equal(np.asarray(new), _out)

        os.remove('out.nii')
"""

if __name__ == '__main__':
    # usage: nosetests -sv test_nifti1.py
    nose.runmodule()
