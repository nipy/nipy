import os
#from numpy import uint8, int16, int32, float32, float64, complex64, array, dtype
from numpy.core.memmap import memmap as memmap_type
import numpy as N

from neuroimaging.utils.odict import odict
#import neuroimaging.data_io.formats.binary
from neuroimaging.data_io import DataSource
import neuroimaging.sandbox.refactoring.formats.binary as bin
import neuroimaging.sandbox.refactoring.formats.analyze as anlz
from neuroimaging.sandbox.refactoring.formats.nifti1_ext import quatern2mat, \
     mat2quatern
from neuroimaging.core.reference.axis import space, spacetime
from neuroimaging.core.reference.mapping import Affine
from neuroimaging.core.reference.grid import SamplingGrid
from neuroimaging.utils.path import path



# datatype is a one bit flag into the datatype identification byte of the
# Analyze header. 
UBYTE = 2
SHORT = 4
INTEGER = 8
FLOAT = 16
COMPLEX = 32
DOUBLE = 64
RGB = 128 # has no translation!
INT8 = 256
UINT16 = 512
UINT32 = 768
INT64 = 1024
UINT64 = 1280
FLOAT128 = 1536 # has no translation!
COMPLEX128 = 1792
COMPLEX256 = 2048 # has no translation!

datatype2sctype = {
    UBYTE: N.uint8,
    SHORT: N.int16,
    INTEGER: N.int32,
    FLOAT: N.float32,
    COMPLEX: N.complex64,
    DOUBLE: N.float64,
    INT8: N.int8,
    UINT16: N.uint16,
    UINT32: N.uint32,
    INT64: N.int64,
    UINT64: N.uint64,
    COMPLEX128: N.complex128,
}

sctype2datatype = dict([(v,k) for k,v in datatype2sctype.items()])

# some bit-mask codes
NIFTI_UNITS_UNKNOWN = 0
NIFTI_UNITS_METER = 1
NIFTI_UNITS_MM = 2
NIFTI_UNITS_MICRON = 3
NIFTI_UNITS_SEC = 8
NIFTI_UNITS_MSEC = 16
NIFTI_UNITS_USEC = 24
NIFTI_UNITS_HZ = 32
NIFTI_UNITS_PPM = 40
NIFTI_UNITS_RADS = 48

unitcode2units = {
    0: '',
    1: 'm',
    2: 'mm',
    3: 'um',
    8: 's',
    16: 'ms',
    24: 'us',
    32: 'hz',
    40: 'ppm',
    48: 'rad',
}
units2unitcode = dict([(v,k) for k,v in unitcode2units.items()])

#q/sform codes
NIFTI_XFORM_UNKNOWN = 0
NIFTI_XFORM_SCANNER_ANAT = 1
NIFTI_XFORM_ALIGNED_ANAT = 2
NIFTI_XFORM_TALAIRACH = 3
NIFTI_XFORM_MNI_152 = 4

#slice codes:
NIFTI_SLICE_UNKNOWN = 0
NIFTI_SLICE_SEQ_INC = 1
NIFTI_SLICE_SEQ_DEC = 2
NIFTI_SLICE_ALT_INC = 3
NIFTI_SLICE_ALT_DEC = 4

# intent codes
NIFTI_INTENT_NONE = 0
NIFTI_INTENT_CORREL = 2
NIFTI_INTENT_TTEST = 3
NIFTI_INTENT_FTEST = 4
NIFTI_INTENT_ZSCORE = 5
NIFTI_INTENT_CHISQ = 6
NIFTI_INTENT_BETA = 7
NIFTI_INTENT_BINOM = 8
NIFTI_INTENT_GAMMA = 9
NIFTI_INTENT_POISSON = 10
NIFTI_INTENT_NORMAL = 11
NIFTI_INTENT_FTEST_NONC = 12
NIFTI_INTENT_CHISQ_NONC = 13
NIFTI_INTENT_LOGISTIC = 14
NIFTI_INTENT_LAPLACE = 15
NIFTI_INTENT_UNIFORM = 16
NIFTI_INTENT_TTEST_NONC = 17
NIFTI_INTENT_WEIBULL = 18
NIFTI_INTENT_CHI = 19
NIFTI_INTENT_INVGAUSS = 20
NIFTI_INTENT_EXTVAL = 21
NIFTI_INTENT_PVAL = 22
NIFTI_INTENT_LOGPVAL = 23
NIFTI_INTENT_LOG10PVAL = 24
NIFTI_FIRST_STATCODE = 2
NIFTI_LAST_STATCODE = 24
NIFTI_INTENT_ESTIMATE = 1001
NIFTI_INTENT_LABEL = 1002
NIFTI_INTENT_NEURONAME = 1003
NIFTI_INTENT_GENMATRIX = 1004
NIFTI_INTENT_SYMMATRIX = 1005
NIFTI_INTENT_DISPVECT = 1006 # specifically for displacements
NIFTI_INTENT_VECTOR = 1007 # for any other type of vector
NIFTI_INTENT_POINTSET = 1008
NIFTI_INTENT_TRIANGLE = 1009
NIFTI_INTENT_QUATERNION = 1010
NIFTI_INTENT_DIMLESS = 1011

# The NIFTI header
HEADER_SIZE = 348
struct_formats = odict((
    ('sizeof_hdr','i'),
    ('data_type','10s'),
    ('db_name','18s'),
    ('extents','i'),
    ('session_error','h'),
    ('regular','c'),
    ('dim_info','c'),
    ('dim','8h'),
    ('intent_p1','f'),
    ('intent_p2','f'),
    ('intent_p3','f'),
    ('intent_code','h'),
    ('datatype', 'h'),
    ('bitpix','h'),
    ('slice_start', 'h'),
    ('pixdim','8f'),
    ('vox_offset','f'),
    ('scl_slope','f'),
    ('scl_inter','f'),
    ('slice_end','h'),
    ('slice_code','c'),
    ('xyzt_units','c'),
    ('cal_max','f'),
    ('cal_min','f'),
    ('slice_duration','f'),
    ('toffset','f'),
    ('glmax','i'),
    ('glmin','i'),
    ('descrip','80s'),
    ('aux_file','24s'),
    ('qform_code','h'),
    ('sform_code','h'),
    ('quatern_b','f'),
    ('quatern_c','f'),
    ('quatern_d','f'),
    ('qoffset_x','f'),
    ('qoffset_y','f'),
    ('qoffset_z','f'),
    ('srow_x','4f'),
    ('srow_y','4f'),
    ('srow_z','4f'),
    ('intent_name','16s'),
    ('magic','4s'),
))
field_formats = struct_formats.values()

##### define an extension here

# an extension is n*16 bytes long, the first 8 bytes are:
# int esize --> the size of the extension in bytes
# int ecode --> the code of the extension


##############################################################################
class Nifti1(bin.BinaryFormat):
    """
    A class to read and write NIFTI format images.
    """

    # Anything which should be default different than field-defaults
    _field_defaults = {'sizeof_hdr': HEADER_SIZE,
                       'scl_slope': 1.0,
                       'magic': 'n+1\x00',
                       'pixdim': [1,0,0,0,0,0,0,0],
                       }

    extensions = ('.img', '.hdr', '.nii', '.mat')
    usematfile = True


    #-------------------------------------------------------------------------
    def __init__(self, filename, mode="r", datasource=DataSource(), **keywords):
        """
        Constructs a Nifti binary format object with at least a filename
        possible additional keyword arguments:
        mode = mode to open the memmap (default is "r")
        datasource = ???
        grid = Grid object
        sctype = numpy scalar type
        intent = meaning of data
        clobber = allowed to clobber?
        usemat = use mat file?
        """

        bin.BinaryFormat.__init__(self, filename, mode, datasource, **keywords)
        self.clobber = keywords.get('clobber', False)
        self.intent = keywords.get('intent', '')
        ### DOES THIS APPLY?
        self.usematfile = keywords.get('usemat', False)
        self.mat_file = self.filebase+".mat"
        ###
        
        self.header_file, self.data_file = self.nifti_filenames()
        # does this need to be redundantly assigned?
        self.header_formats = struct_formats

        # fill the header dictionary in order, with any default values
        self.header_defaults()
        if self.mode[0] is "w":
            # should try to populate the canonical fields and
            # corresponding header fields with info from grid?
            self.sctype = keywords.get('sctype', N.float64)
            self.byteorder = bin.NATIVE
            if self.grid is not None:
                self.header_from_given()
            else:
                raise NotImplementedError("Don't know how to dcreate header info without a grid object")
            self.write_header()
        else:
            # this should work
            self.byteorder = anlz.Analyze.guess_byteorder(self.header_file)
            self.read_header()
            self.sctype = datatype2sctype[self.header['datatype']]
            self.ndim = self.header['dim'][0]

        # fill in the canonical list as best we can for Analyze
        #self.inform_canonical()

        ########## This could stand a clean-up ################################
        if self.grid is None:
            
            if self.usematfile:
                self.grid.transform(self.read_mat())
                # assume .mat matrix uses FORTRAN indexing
                self.grid = self.grid.matlab2python()
            else:
                origin = (self.header['qoffset_x'],
                          self.header['qoffset_y'],
                          self.header['qoffset_z'])
                step = tuple(self.header['pixdim'][1:4])
                shape = tuple(self.header['dim'][1:4])
                if self.ndim == 3:
                    axisnames = space[::-1]
                elif self.ndim == 4 and self.nvector <= 1:
                    axisnames = spacetime[::-1]
                    origin = origin + (1,)
                    step = step + (self.header['pixdim'][5],)
                    shape = shape + (self.header['dim'][5],)
##                     if self.squeeze:
##                     if self.dim[4] == 1:
##                         origin = origin[0:3]
##                         step = step[0:3]
##                         axisnames = axisnames[0:3]
##                         shape = self.dim[1:4]
                elif self.ndim == 4 and self.nvector > 1:
                    axisnames = ('vector_dimension', ) + space[::-1]
                    origin = (1,) + origin
                    step = (1,) + step
                    shape = shape + (self.header['dim'][5],)
##                     if self.squeeze:
##                         if self.dim[1] == 1:
##                             origin = origin[1:4]
##                             step = step[1:4]
##                             axisnames = axisnames[1:4]
##                             shape = self.dim[2:5]

                self.grid = SamplingGrid.from_start_step(names=axisnames,
                                                shape=shape,
                                                start=-N.array(origin)*step,
                                                step=step)
                t = self.transform()
                self.grid.mapping.transform[:3,:3] = t[:3,:3]
                self.grid.mapping.transform[:3,-1] = t[:3,-1]
                ### why is this here?
                self.grid = self.grid.matlab2python()
        else:
            self.grid = grid
            
        self.attach_data(offset=int(self.header['vox_offset']))


    def nifti_filenames(self):
        # Nifti single file will be the preferred type for creation
##         if self.mode[0] == "w":
##             return (self.filebase+".nii",self.filebase+".nii")
        return os.path.exists(self.filebase+".hdr") and \
               (self.filebase+".hdr", self.filebase+".img") or\
               (self.filebase+".nii", self.filebase+".nii")
    
    #-------------------------------------------------------------------------
    @staticmethod
    def _default_field_value(fieldname, fieldformat):
        "[STATIC] Get the default value for the given field."
        return Nifti1._field_defaults.get(fieldname, None) or \
               bin.format_defaults[fieldformat[-1]]
    
    #-------------------------------------------------------------------------
    def header_defaults(self):
        for field,format in self.header_formats.items():
            self.header[field] = self._default_field_value(field,format)


    def header_from_given(self):
        # try to set up these fields from what we know:
        # datatype
        # bitpix
        # quatern_b,c,d
        # qoffset_x,y,z
        # qfac
        # bitpix
        # dim

        
        self.grid = self.grid.python2matlab()
        self.header['datatype'] = sctype2datatype[self.sctype]
        self.header['bitpix'] = N.dtype(self.sctype).itemsize
        self.ndim = len(self.grid.shape)
    
        if not isinstance(self.grid.mapping, Affine):
            raise NIFTI1FormatError, 'error: non-Affine grid in writing out NIFTI-1 file'

        ddim = self.grid.ndim - 3
        t = self.grid.mapping.transform[ddim:,ddim:]

        qb, qc, qd, qx, qy, qz, dx, dy, dz, qfac = quaternion(t)

        (self.header['quatern_b'],
         self.header['quatern_c'],
         self.header['quatern_d']) = qb, qc, qd
        
        (self.header['qoffset_x'],
         self.header['qoffset_y'],
         self.header['qoffset_z']) = qx, qy, qz

        _pixdim = [0.]*8
        _pixdim[0:4] = [qfac, dx, dy, dz]
        self.header['pixdim'] = _pixdim

        self.qform_code = 1
        
        self.header['dim'] = \
                        [self.ndim] + list(self.grid.shape) + [0]*(8-self.ndim)

        self.grid = self.grid.matlab2python()
        
    def transform(self):
        """
        Return 4x4 transform matrix based on the NIFTI attributes
        for the 3d (spatial) part of the mapping.
        If self.sform_code > 0, use the attributes srow_{x,y,z}, else
        if self.qform_code > 0, use the quaternion
        else use a diagonal matrix filled in by pixdim.

        See help(neuroimaging.data_io.formats.nifti1_ext) for explanation.

        """

        qfac = float(self.header['pixdim'][0])
        if qfac not in [-1.,1.]:
            raise NIFTI1FormatError('invalid qfac: orientation unknown')
        
        value = N.zeros((4,4), N.float64)
        value[3,3] = 1.0
        
        if self.header['qform_code'] > 0:
            
            value = quatern2mat(b=self.header['quatern_b'],
                              c=self.header['quatern_c'],
                              d=self.header['quatern_d'],
                              qx=self.header['qoffset_x'],
                              qy=self.header['qoffset_y'],
                              qz=self.header['qoffset_z'],
                              dx=self.header['pixdim'][1],
                              dy=self.header['pixdim'][2],
                              dz=self.header['pixdim'][3],
                              qfac=qfac)

        elif self.header['sform_code'] > 0:

            value[0] = N.array(self.header['srow_x'])
            value[1] = N.array(self.header['srow_y'])
            value[2] = N.array(self.header['srow_z'])

        return value


    def postread(self, x):
        """
        NIFTI-1 normalization based on scl_slope and scl_inter.
        """
        return x * self.header['scl_slope'] + self.header['scl_inter']

    def prewrite(self, x):
        """
        NIFTI-1 normalization based on scl_slope and scl_inter.
        """
        return (x - self.header['scl_inter']) / self.header['scl_slope']
        

if __name__=='__main__':
    import sys
    sys.path.append(os.path.abspath(__file__))
    import pdb
    #fname = 'sagittal_gems_TEM1.recon.nii'
    fname = 'newNiftFile'
    pdb.run('Nifti1(fname,mode=\'w\',)')
