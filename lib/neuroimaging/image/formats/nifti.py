import struct, os, sys, types
import numpy as N

from neuroimaging.data import iszip, unzip, DataSource
from neuroimaging.reference.axis import space, spacetime
from neuroimaging.reference.mapping import Affine, Mapping
from neuroimaging.reference.grid import SamplingGrid

from validators import BinaryHeaderAtt, BinaryFile, traits

# NIFTI-1 constants

DT_NONE = 0
DT_UNKNOWN = 0 # what it says, dude
DT_BINARY = 1 # binary (1 bit/voxel)
DT_UNSIGNED_CHAR = 2 # unsigned char (8 bits/voxel)
DT_SIGNED_SHORT = 4 # signed short (16 bits/voxel)
DT_SIGNED_INT = 8 # signed int (32 bits/voxel)
DT_FLOAT = 16 # float (32 bits/voxel)
DT_COMPLEX = 32 # complex (64 bits/voxel)
DT_DOUBLE = 64 # double (64 bits/voxel)
DT_RGB = 128 # RGB triple (24 bits/voxel)
DT_ALL = 255 # not very useful (?)
DT_UINT8 = 2
DT_INT16 = 4
DT_INT32 = 8
DT_FLOAT32 = 16
DT_COMPLEX64 = 32
DT_FLOAT64 = 64
DT_RGB24 = 128
DT_INT8 = 256 # signed char (8 bits)
DT_UINT16 = 512 # unsigned short (16 bits)
DT_UINT32 = 768 # unsigned int (32 bits)
DT_INT64 = 1024 # long long (64 bits)
DT_UINT64 = 1280 # unsigned long long (64 bits)
DT_FLOAT128 = 1536 # long double (128 bits)
DT_COMPLEX128 = 1792 # double pair (128 bits)
DT_COMPLEX256 = 2048 # long double pair (256 bits)
NIFTI_TYPE_UINT8 = 2
NIFTI_TYPE_INT16 = 4
NIFTI_TYPE_INT32 = 8
NIFTI_TYPE_FLOAT32 = 16
NIFTI_TYPE_COMPLEX64 = 32
NIFTI_TYPE_FLOAT64 = 64
NIFTI_TYPE_RGB24 = 128
NIFTI_TYPE_INT8 = 256
NIFTI_TYPE_UINT16 = 512
NIFTI_TYPE_UINT32 = 768
NIFTI_TYPE_INT64 = 1024
NIFTI_TYPE_UINT64 = 1280
NIFTI_TYPE_FLOAT128 = 1536
NIFTI_TYPE_COMPLEX128 = 1792
NIFTI_TYPE_COMPLEX256 = 2048
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
NIFTI_XFORM_UNKNOWN = 0
NIFTI_XFORM_SCANNER_ANAT = 1
NIFTI_XFORM_ALIGNED_ANAT = 2
NIFTI_XFORM_TALAIRACH = 3
NIFTI_XFORM_MNI_152 = 4
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
NIFTI_SLICE_UNKNOWN = 0
NIFTI_SLICE_SEQ_INC = 1
NIFTI_SLICE_SEQ_DEC = 2
NIFTI_SLICE_ALT_INC = 3
NIFTI_SLICE_ALT_DEC = 4
NIFTI_SLICE_ALT_INC2 = 5 # 05 May 2005: RWCox
NIFTI_SLICE_ALT_DEC2 = 6 # 05 May 2005: RWCox

DT = [DT_NONE, DT_UNKNOWN, DT_BINARY, DT_UNSIGNED_CHAR, DT_SIGNED_SHORT, DT_SIGNED_INT, DT_FLOAT, DT_COMPLEX, DT_DOUBLE, DT_RGB, DT_ALL, DT_UINT8, DT_INT16, DT_INT32, DT_FLOAT32, DT_COMPLEX64, DT_FLOAT64, DT_RGB24, DT_INT8, DT_UINT16, DT_UINT32, DT_INT64, DT_UINT64, DT_FLOAT128, DT_COMPLEX128, DT_COMPLEX256]

NIFTI_TYPE = [NIFTI_TYPE_UINT8, NIFTI_TYPE_INT16, NIFTI_TYPE_INT32, NIFTI_TYPE_FLOAT32, NIFTI_TYPE_COMPLEX64, NIFTI_TYPE_FLOAT64, NIFTI_TYPE_RGB24, NIFTI_TYPE_INT8, NIFTI_TYPE_UINT16, NIFTI_TYPE_UINT32, NIFTI_TYPE_INT64, NIFTI_TYPE_UINT64, NIFTI_TYPE_FLOAT128, NIFTI_TYPE_COMPLEX128, NIFTI_TYPE_COMPLEX256]

NIFTI_INTENT = [NIFTI_INTENT_NONE, NIFTI_INTENT_CORREL, NIFTI_INTENT_TTEST, NIFTI_INTENT_FTEST, NIFTI_INTENT_ZSCORE, NIFTI_INTENT_CHISQ, NIFTI_INTENT_BETA, NIFTI_INTENT_BINOM, NIFTI_INTENT_GAMMA, NIFTI_INTENT_POISSON, NIFTI_INTENT_NORMAL, NIFTI_INTENT_FTEST_NONC, NIFTI_INTENT_CHISQ_NONC, NIFTI_INTENT_LOGISTIC, NIFTI_INTENT_LAPLACE, NIFTI_INTENT_UNIFORM, NIFTI_INTENT_TTEST_NONC, NIFTI_INTENT_WEIBULL, NIFTI_INTENT_CHI, NIFTI_INTENT_INVGAUSS, NIFTI_INTENT_EXTVAL, NIFTI_INTENT_PVAL, NIFTI_INTENT_LOGPVAL, NIFTI_INTENT_LOG10PVAL, NIFTI_INTENT_ESTIMATE, NIFTI_INTENT_LABEL, NIFTI_INTENT_NEURONAME, NIFTI_INTENT_GENMATRIX, NIFTI_INTENT_SYMMATRIX, NIFTI_INTENT_DISPVECT, NIFTI_INTENT_VECTOR, NIFTI_INTENT_POINTSET, NIFTI_INTENT_TRIANGLE, NIFTI_INTENT_QUATERNION, NIFTI_INTENT_DIMLESS]

NIFTI_XFORM = [NIFTI_XFORM_UNKNOWN, NIFTI_XFORM_SCANNER_ANAT, NIFTI_XFORM_ALIGNED_ANAT, NIFTI_XFORM_TALAIRACH, NIFTI_XFORM_MNI_152]

NIFTI_UNITS = [NIFTI_UNITS_UNKNOWN, NIFTI_UNITS_METER, NIFTI_UNITS_MM, NIFTI_UNITS_MICRON, NIFTI_UNITS_SEC, NIFTI_UNITS_MSEC, NIFTI_UNITS_USEC, NIFTI_UNITS_HZ, NIFTI_UNITS_PPM, NIFTI_UNITS_RADS]

NIFTI_SLICE = [NIFTI_SLICE_UNKNOWN, NIFTI_SLICE_SEQ_INC, NIFTI_SLICE_SEQ_DEC, NIFTI_SLICE_ALT_INC, NIFTI_SLICE_ALT_DEC, NIFTI_SLICE_ALT_INC2, NIFTI_SLICE_ALT_DEC2]

# NIFTI-1 datatypes

datatypes = {DT_NONE:None, # fail if unknown
             DT_UNKNOWN:None, 
             DT_BINARY:N.bool8,
             DT_UNSIGNED_CHAR:N.uint8,
             DT_SIGNED_SHORT:N.int16,
             DT_SIGNED_INT:N.int32,
             DT_FLOAT:N.float32,
             DT_COMPLEX:None,
             DT_DOUBLE:N.float64,
             DT_RGB:None,
             DT_ALL:None,
             DT_UINT8:N.uint8,
             DT_INT16:N.int16,
             DT_INT32:N.int32,
             DT_FLOAT32:N.float32,
             DT_COMPLEX64:N.complex64,
             DT_FLOAT64:N.float64,
             DT_RGB24:None,
             DT_INT8:N.int8,
             DT_UINT16:N.uint16,
             DT_UINT32:N.uint32,
             DT_INT64:N.int64,
             DT_UINT64:N.uint64,
             DT_FLOAT128:None,
             DT_COMPLEX128:None,
             DT_COMPLEX256:None}


_byteorder_dict = {'big':'>', 'little':'<'}


dimorder = ['xspace', 'yspace', 'zspace', 'time', 'vector_dimension']

class NIFTI(BinaryFile):
    """
    A class that implements the nifti1 header with some typechecking.
    NIFTI-1 attributes must conform to their description in nifti1.h.

    You MUST pass the HEADER file as the filename, unlike ANALYZE where either will do.
    This may be a .hdr file ('ni1' case) or a .nii file ('n+1') case. The code needs to
    have the header to figure out what kind of file it is.
    """

    sizeof_hdr = BinaryHeaderAtt('i', 0, 348)
    data_type = BinaryHeaderAtt('10s', 4, ' '*10)
    db_name = BinaryHeaderAtt('18s', 14, ' '*18)
    extents = BinaryHeaderAtt('i', 32, 0)
    session_error = BinaryHeaderAtt('h', 36, 0)
    regular = BinaryHeaderAtt('s', 38, 'r')
    dim_info = BinaryHeaderAtt('b', 39, 0)
    dim = BinaryHeaderAtt('8h', 40, (4,1,1,1,1) + (0,)*3)
    intent_p1 = BinaryHeaderAtt('f', 56, 0.)
    intent_p2 = BinaryHeaderAtt('f', 60, 0.)
    intent_p3 = BinaryHeaderAtt('f', 64, 0.)
    intent_code = BinaryHeaderAtt('h', 68, 0)
    datatype = BinaryHeaderAtt('h', 70, 0)
    bitpix = BinaryHeaderAtt('h', 72, 0)
    slice_start = BinaryHeaderAtt('h', 74, 0)
    pixdim = BinaryHeaderAtt('8f', 76, (1.,) + (0.,)*7)
    vox_offset = BinaryHeaderAtt('f', 108, 0)
    scl_slope = BinaryHeaderAtt('f', 112, 1.0)
    scl_inter = BinaryHeaderAtt('f', 116, 0.)
    slice_end = BinaryHeaderAtt('h', 120, 0)
    slice_code = BinaryHeaderAtt('b', 122, 0)
    xyzt_units = BinaryHeaderAtt('b', 123, 0)
    cal_max = BinaryHeaderAtt('f', 124, 0)
    cal_min = BinaryHeaderAtt('f', 128, 0)
    slice_duration = BinaryHeaderAtt('f', 132, 0)
    toffset = BinaryHeaderAtt('f', 136, 0)
    glmax = BinaryHeaderAtt('i', 140, 0)
    glmin = BinaryHeaderAtt('i', 144, 0)
    descrip = BinaryHeaderAtt('80s', 148, ' '*80)
    aux_file = BinaryHeaderAtt('24s', 228, ' '*24)
    qform_code = BinaryHeaderAtt('h', 252, 0)
    sform_code = BinaryHeaderAtt('h', 254, 0)
    quatern_b = BinaryHeaderAtt('f', 256, 0.0)
    quatern_c = BinaryHeaderAtt('f', 260, 0.)
    quatern_d = BinaryHeaderAtt('f', 264, 0.)
    qoffset_x = BinaryHeaderAtt('f', 268, 0.)
    qoffset_y = BinaryHeaderAtt('f', 272, 0.)
    qoffset_z = BinaryHeaderAtt('f', 276, 0.)
    srow_x = BinaryHeaderAtt('4f', 280, [0.,0.,1.,0.])
    srow_y = BinaryHeaderAtt('4f', 296, [0.,1.,0.,0.])
    srow_z = BinaryHeaderAtt('4f', 312, [1.,0.,0.,0.])
    intent_name = BinaryHeaderAtt('16s', 328, ' '*16)
    magic = BinaryHeaderAtt('4s', 344, 'ni1\0')

    extensions = traits.Trait(['.img', '.hdr', '.nii'], desc='Extensions supported by this format.')

    def __init__(self, hdrfilename, mode='r', create=False, datasource=DataSource(), **keywords):
        BinaryFile.__init__(self, **keywords)
        self.datasource = datasource
        ext = os.path.splitext(hdrfilename)[1]
        if ext not in ['.nii', '.hdr']:
            raise ValueError, 'NIFTI images need .hdr or .nii file specified.'

        self.filebase, self.fileext = os.path.splitext(hdrfilename)
        self.hdrfilename = hdrfilename
        
        # figure out machine byte order -- needed for reading binary data

        self.byteorder = sys.byteorder
        self.bytesign = _byteorder_dict[self.byteorder]

        self.readheader()
        
    def check_byteorder(self, hdrfile):
        """
        A check of byteorder based on the 'sizeof_hdr' attribute,
        which should equal 348.
        """

        sizeof_hdr = self.trait('sizeof_hdr')
        sizeof_hdr.handler.bytesign = self.bytesign
        value = sizeof_hdr.handler.read(hdrfile)

        if value != 348:
            if self.bytesign in ['>', '!']:
                self.bytesign = '<'
                self.byteorder = 'little'
            else:
                self.bytesign = '!'
                self.byteorder = 'big'
        hdrfile.seek(0,0)
        
    def readheader(self, hdrfilename=None):
        """
        Read in a NIFTI-1 header file, filling all default values.
        """

        hdrfilename = hdrfilename or self.hdrfilename
        hdrfile = self.datasource.open(hdrfilename)

        self.check_byteorder(hdrfile)
        BinaryFile.readheader(self, hdrfile)

        self.dtype = datatypes[self.datatype]
        hdrfile.close()

##         if self.magic == 'n+1\x00':
##             self.brikfile = self.hdrfile
##             self.offset = self.vox_offset # should be 352 for most such files
##         else:
##             if mode in ['r']:
##                 self.brikfile = file(self.filebase + '.img', mode=mode)
##             elif mode in ['r+', 'w'] and self.clobber and create:
##                 self.brikfile = file(self.filebase + '.img', mode=mode)
##             elif mode in ['r+', 'w'] and self.clobber and not create:
##                 self.brikfile = file(self.filebase + '.img', mode='rb+')
##             self.offset = 0

##         self.ndim = self.dim[0]
##         self.shape = self.dim[1:(1+self.ndim)][::-1]
##         self.step = self.pixdim[1:(1+self.ndim)][::-1]
##         self.start = [0.] * self.ndim

##         ## Setup affine transformation

##         self.incoords = Coordinates.VoxelCoordinates('voxel', self.indim)
##         self.outcoords = Coordinates.OrthogonalCoordinates('world', self.outdim)

##         matrix = self._transform()
##         self.mapping = Mapping.Affine(self.incoords, self.outcoords, matrix)
##         if NIFTI.reorder_xfm:
##             self.mapping.reorder(reorder_dims=NIFTI.reorder_dims)

##         self.incoords = self.mapping.input_coords
##         self.outcoords = self.mapping.output_coords

##         self.start = self.mapping.output_coords.start
##         self.step = self.mapping.output_coords.step
##         self.shape = self.mapping.output_coords.shape

##     def _transform(self):
##         """
##         Return 4x4 transform matrix based on the NIFTI attributes.
##         If self.sform_code > 0, use the attributes srow_{x,y,z}, else
##         use the quaternion. The calculation is taken from
        
##         http://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/quatern.html


##         """

##         value = N.zeros((4,4), N.float64)
##         value[3,3] = 1.0
        
##         if self.sform_code > 0:

##             value[0] = self.srow_x
##             value[1] = self.srow_y
##             value[2] = self.srow_z

##         elif self.qform_code > 0:
            
##             a, b, c, d = (1.0, self.quatern_b, self.quatern_c, self.quatern_d)
##             R = N.array([[a*a+b*b-c*c-d*d, 2.*b*c-2*a*d,2*b*d+2*a*c],
##                                 [2*b*c+2*a*d, a*a+c*c-b*b-d*d, 2*c*d-2*a*b],
##                                 [2*b*d-2*a*c, 2*c*d+2*a*b, a*a+d*d-c*c-b*b]])
##             if self.pixdim[0] == 0.0:
##                 qfac = 1.0
##             else:
##                 qfac = self.pixdim[0]
##             R[:,2] = qfac * R[:,2]

##             value[0:3,0:3] = R
##             value[0,3] = self.qoffset_x
##             value[1,3] = self.qoffset_y
##             value[2,3] = self.qoffset_z

##         else:
##             value[0,0] = self.pixdim[1]
##             value[1,1] = self.pixdim[2]
##             value[2,2] = self.pixdim[3]

##         return value
            
##     def _dimensions2dim(self, dimensions):
##         '''This routine tries to a list of dimensions into sensible NIFTI dimensions.'''

##         _dimnames = [dim.name for dim in dimensions]
##         _dimshape = [dim.length for dim in dimensions]
##         _dimdict = {}

##         for _name in _dimnames:
##             _dimdict[_name] = dimensions[_dimnames.index(_name)]
            
##         if 'vector_dimension' in _dimnames:
##             ndim = 5
##             has_vector = True
##         else:
##             has_vector = False
##             if 'time' in _dimnames:
##                 ndim = 4
##                 has_time = True
##             else:
##                 has_time = False
##                 ndim = len(dimensions)

##         dim = [ndim]
##         pixdim = list(self.pixdim[0:1])

##         self.spatial_dimensions = []

##         i = 1
##         for _name in ['xspace', 'yspace', 'zspace']:
##             try: # see if these dimensions exist
##                 dim.append(_dimdict[_name].length)
##                 pixdim.append(abs(_dimdict[_name].step))
##                 self.spatial_dimensions.append(_dimdict[_name])
##             except: # else set pixdim=0 even though dimension may be needed
##                 dim.append(1)
##                 pixdim.append(0.)
##         if has_time and not has_vector:
##             dim.append(_dimdict['time'].length)
##             pixdim.append(abs(_dimdict['time'].step))
##         elif not has_time and has_vector:
##             dim.append(1)
##             pixdim.append(0.)
##             dim.append(_dimdict['vector_dimension'].length)
##         elif has_time and has_vector:
##             dim.append(_dimdict['time'].length)
##             pixdim.append(abs(_dimdict['time'].step))
##             dim.append(_dimdict['vector_dimension'].length)

##         self.outdim = dimensions
##         self.indim = [Dimension.RegularDimension(name=outdim.name, length=outdim.length, start=0.0, step=1.0) for outdim in self.outdim]

##         self.dim = tuple(dim + [1] * (8 - len(dim)))
##         self.pixdim = tuple(pixdim + [0.] * (8 - len(pixdim)))
        
##     def read(self, start, count, **keywords):
##         return_value = Utils.brickutils.readbrick(self.brikfile, start, count, self.shape, byteorder=self.byteorder, intype = self.typecode, offset=self.offset)
##         if self.scl_slope not in  [1.0, 0.0]:
##             return_value = self.scl_slope * return_value
##         if self.scl_inter != 0.0:
##             return_value = return_value + self.scl_inter
##         return return_value

##     def write(self, start, data, **keywords):
##         self.close()
##         self.open(mode='r+', header=False)
##         if self.scl_inter != 0:
##             outdata = data - self.scl_inter
##         else:
##             outdata = data
##         if self.scl_slope != 1.0:
##             outdata = outdata / self.scl_slope
##         if len(start) == 3 and len(self.shape) == 4 and self.shape[0] == 1:
##             newstart = (0,) + tuple(start) # Is the NIFTI file "really 3d"?
##         else:
##             newstart = start
##         Utils.brickutils.writebrick(self.brikfile, newstart, outdata, self.shape, byteorder = self.byteorder, outtype = self.typecode, offset = self.offset)
##         return 

##     def close(self, header=True, brick=True):
##         if header:
##             self.hdrfile.close()
##         if brick:
##             self.brikfile.close()

##     def open(self, mode='r', header=True, brick=True):
##         if mode != 'r' and not self.clobber:
##             raise ValueError, 'clobber does not agree with mode'
##         if mode is None:
##             mode = 'r'
##         if mode == 'r':
##             mode = 'rb'
##         if mode == 'w':
##             mode = 'wb'
##         elif mode == 'r+':
##             mode = 'rb+'
##         if header:
##             try:
##                 self.hdrfile = file(self.hdrfile.name, mode=mode)
##                 self.hdrfile.seek(0,0)
##             except:
##                 raise ValueError, 'errors opening header file %s' % self.hdrfile.name
##         if brick:
##             try:
##                 self.brikfile = file(self.brikfile.name, mode=mode)
##                 self.brikfile.seek(0,0)
##             except:
##                 raise ValueError, 'errors opening data file %s' % self.hdrfile.name



##     def readheader(self):
##         self.close(brick=False)
##         self.open(mode='r', brick=False)
##         try:
##             self.hdrfile.seek(0,0)
##         except:
##             self.hdrfile = file(self.hdrfile.name)
##             self.hdrfile.seek(0,0)
##         for att in _header_atts:
##             tmp = self.hdrfile.read(struct.calcsize(self.bytesign + att[1]))
##             value = struct.unpack(self.bytesign + att[1], tmp)
##             if len(value) == 1:
##                 setattr(self, att[0], value[0])
##             else:
##                 setattr(self, att[0], list(value))

##         self.close(brick=False)

##         dimensions = []
##         self.ndim = self.dim[0]
##         for i in range(self.ndim):
##             if self.pixdim[i+1] != 0:
##                 dimensions.append(Dimension.RegularDimension(name=dimorder[i], length=self.dim[i+1], start=0.0, step=self.pixdim[i+1]))
##         self._dimensions2dim(dimensions)
##         return

##     def writeheader(self, hdrfile = None):
##         if not hdrfile and self.clobber:
##             hdrfile = file(self.hdrfile.name, 'w')
##         elif self.clobber:
##             self.hdrfile.close()
##             self.hdrfile = file(self.hdrfile.name, 'w')
##             hdrfile = self.hdrfile
##         else:
##             raise ValueError, 'clobber is False and no hdrfile supplied'
##         for att in _header_atts: # Fill in default values if attributes are not present
##             if not hasattr(self, att[0]):
##                 setattr(self, att[0], att[3])
##         for att in _atts:
##             value = getattr(self, att[0])
##             if att[1][-1] == 's':
##                 value = value.__str__()
##             if not att[2]:
##                 value = (value,)
##             hdrfile.write(apply(struct.pack, (self.bytesign + att[1],) + tuple(value)))
##         hdrfile.close()

##     def __str__(self):
##         value = ''
##         for att in _header_atts:
##             _value = getattr(self, att[0])
##             value = value + '%s:%s=%s\n' % (os.path.split(self.hdrfilename)[1], att[0], _value.__str__())
##         return value[:-1]


"""
URLPipe class expects this.
"""

creator = NIFTI

