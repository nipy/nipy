__docformat__ = 'restructuredtext'

import numpy as np

from neuroimaging.utils.odict import odict
from neuroimaging.data_io.datasource import DataSource
from neuroimaging.data_io.formats import utils, binary, analyze
from neuroimaging.data_io.formats.nifti1_ext import quatern2mat, \
     mat2quatern

from neuroimaging.core.reference.mapping import Affine
from neuroimaging.core.reference.grid import SamplingGrid


class Nifti1FormatError(Exception):
    """
    Nifti format error exception
    """

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
    UBYTE: np.uint8,
    SHORT: np.int16,
    INTEGER: np.int32,
    FLOAT: np.float32,
    COMPLEX: np.complex64,
    DOUBLE: np.float64,
    INT8: np.int8,
    UINT16: np.uint16,
    UINT32: np.uint32,
    INT64: np.int64,
    UINT64: np.uint64,
    COMPLEX128: np.complex128,
}

sctype2datatype = dict([(v, k) for k, v in datatype2sctype.items()])

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
    NIFTI_UNITS_UNKNOWN: '',
    NIFTI_UNITS_METER: 'm',
    NIFTI_UNITS_MM: 'mm',
    NIFTI_UNITS_MICRON: 'um',
    NIFTI_UNITS_SEC: 's',
    NIFTI_UNITS_MSEC: 'ms',
    NIFTI_UNITS_USEC: 'us',
    NIFTI_UNITS_HZ: 'hz',
    NIFTI_UNITS_PPM: 'ppm',
    NIFTI_UNITS_RADS: 'rad',
}
units2unitcode = dict([(v, k) for k, v in unitcode2units.items()])

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
    ('dim_info','B'),
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
    ('slice_code','B'),
    ('xyzt_units','B'),
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
    ('qfac','f'),
))
field_formats = struct_formats.values()

##### define an extension here

# an extension is n*16 bytes long, the first 8 bytes are:
# int esize --> the size of the extension in bytes
# int ecode --> the code of the extension

# Anything which should be default different than field-defaults
_field_defaults = {'sizeof_hdr': HEADER_SIZE,
                   'scl_slope': 1.0,
                   'magic': 'n+1\x00',
                   'pixdim': [1,0,0,0,0,0,0,0],
                   'vox_offset': 352.0,
                   }

def _default_field_value(fieldname, fieldformat):
    """ Get the default value for the given field."""

    dflt_val = _field_defaults.get(fieldname, None)
    if dflt_val is not None:
        # Use the default for this field if there is one
        return dflt_val

    multiple_values = fieldformat[:-1]
    a_string = fieldformat[-1] is 's'
    if multiple_values and not a_string:
        # build a list of the default values for this format type
        # ex: '4f' => [0.0, 0.0, 0.0, 0.0]
        val = utils.format_defaults[fieldformat[-1]]
        numvals = int(fieldformat[:-1])
        return [val] * numvals
    else:
        # return single default value for this format type
        return utils.format_defaults[fieldformat[-1]]

def create_default_header(header_formats=None):
    """Create a Nifti1 header with default values.

    BUG: Should fill in dim and pixdim with 1.0 instead of 0.0.
        qfac defaults to 0.0 also and should probably be 1.0

    Parameters
    ----------
    header_formats : {dict}
        Dictionary describing the struct format codes for each key.

    Returns
    -------
    header : {odict}
       An ordered dictionary with default values for each header key.

    Examples
    --------
    
    >>> from neuroimaging.data_io.formats import nifti1
    >>> default_nifti_header = nifti1.create_default_header()

    """

    hdr = odict()
    if header_formats is None:
        fmts = struct_formats.copy()
    else:
        fmts = header_formats
    for field, format in fmts.items():
        hdr[field] = _default_field_value(field, format)
    return hdr

def scale_data(data, **kwargs):
    """Apply scaling to the data.

    Parameters
    ----------
    data : array
        Unscaled data
    kwargs : dict
        dictionary of keyword args used in the intensity scaling for this
        format.  Nifti will look for scale_factor and scale_inter.

    Returns
    -------
    scaled_data : array

    Examples
    --------
    >>> # Scale data from full int16 range to [0,1]
    >>> import numpy as np
    >>> from neuroimaging.data_io.formats import nifti1
    >>> data = np.array([-32768, 0, 32767], dtype=np.int16)
    >>> fac = 1.0/(2**16)
    >>> intr = 0.5
    >>> scaled_data =nifti1.scale_data(data, scale_factor=fac, scale_inter=intr)

    """
    
    scale_factor = kwargs.get('scale_factor', 0.0)
    scale_inter = kwargs.get('scale_inter', 0.0)
    # Only apply if scl_slope is nonzero
    if scale_factor != 0.0:
        return data * scale_factor + scale_inter
    else:
        return data

class Nifti1(binary.BinaryFormat):
    """
    A class to read and write NIFTI format images.
    """

    extensions = ('.img', '.hdr', '.nii', '.mat')

    extendable = False

    def __init__(self, filename, mode="r", datasource=DataSource(), 
                 use_memmap=True, **keywords):
        """
        Constructs a Nifti binary format object with at least a filename
        possible additional keyword arguments:
        mode = mode to open the memmap (default is "r")
        datasource = ???
        grid = Grid object
        dtype = numpy data type
        intent = meaning of data
        clobber = allowed to clobber?
        """

        binary.BinaryFormat.__init__(self, filename, mode, datasource, 
                                     **keywords)
        self.intent = keywords.get('intent', '')

        # Initialize header with default values from nifti1 spec
        self.header_formats = struct_formats.copy()
        self.header = create_default_header(self.header_formats)

        if self.mode[0] is "w":
            # should try to populate the canonical fields and
            # corresponding header fields with info from grid?
            self.byteorder = utils.NATIVE
            self.dtype = np.dtype(keywords.get('dtype', np.float64))
            self.dtype = self.dtype.newbyteorder(self.byteorder)
            if self.grid is not None:
                self._header_from_grid()
            else:
                raise NotImplementedError("Don't know how to create header" \
                                          "info without a grid object")
            self.write_header(clobber=self.clobber)
        else:
            # this should work
            self.byteorder = analyze.Analyze.guess_byteorder(self.header_file,
                                                    datasource=self.datasource)
            self.read_header()
            # we may THINK it's a Nifti, but ...
            if self.header['magic'] not in ('n+1\x00', 'ni1\x00'):
                raise Nifti1FormatError
            tmpsctype = datatype2sctype[self.header['datatype']]
            tmpstr = np.dtype(tmpsctype)
            self.dtype = tmpstr.newbyteorder(self.byteorder)
            self.ndim = self.header['dim'][0]

        if self.grid is None:
            self._grid_from_header()
        
        self.attach_data(offset=int(self.header['vox_offset']), 
                         use_memmap=use_memmap)

    def _get_filenames(self):
        # Nifti single file will be the preferred type for creation
        return self.datasource.exists(self.filebase+".hdr") and \
               (self.filebase+".hdr", self.filebase+".img") or\
               (self.filebase+".nii", self.filebase+".nii")
    def _getaffine_method1(self):
        """
        Method to get image orientation location
        based on Method1 in nifti.h

        METHOD 1 (the "old" way, used only when qform_code = 0):
        -------------------------------------------------------
        The coordinate mapping from (i,j,k) to (x,y,z) is the ANALYZE
        7.5 way.  This is a simple scaling relationship:
        
        x = pixdim[1] * i
        y = pixdim[2] * j
        z = pixdim[3] * k
        
        No particular spatial orientation is attached to these (x,y,z)
        coordinates.  (NIFTI-1 does not have the ANALYZE 7.5 orient field,
        which is not general and is often not set properly.)  This method
        is not recommended, and is present mainly for compatibility with
        ANALYZE 7.5 files.
        

        Returns:
        ________________________
        transmatrix :numpy.array
             simple 4X4 transformation matrix
        """
        origin = (self.header['qoffset_x'],
                  self.header['qoffset_y'],
                  self.header['qoffset_z'])
        step = np.ones(4)
        step[0:4] = self.header['pixdim'][1:5]
        transmatrix = np.eye(4) * step
        transmatrix[:3,3] = origin
        return transmatrix

    def _getaffine_method2(self):
        """
        Method to get image orientation location
        based on Method2 in nifti.h


        METHOD 2 (used when qform_code > 0, which should be the "normal" case):
        ---------------------------------------------------------------------
        The (x,y,z) coordinates are given by the pixdim[] scales, a rotation
        matrix, and a shift.  This method is intended to represent
        "scanner-anatomical" coordinates, which are often embedded in the
        image header (e.g., DICOM fields (0020,0032), (0020,0037), (0028,0030),
        and (0018,0050)), and represent the nominal orientation and location of
        the data.  This method can also be used to represent "aligned"
        coordinates, which would typically result from some post-acquisition
        alignment of the volume to a standard orientation (e.g., the same
        subject on another day, or a rigid rotation to true anatomical
        orientation from the tilted position of the subject in the scanner).
        The formula for (x,y,z) in terms of header parameters and (i,j,k) is:
        
        [ x ]   [ R11 R12 R13 ] [        pixdim[1] * i ]   [ qoffset_x ]
        [ y ] = [ R21 R22 R23 ] [        pixdim[2] * j ] + [ qoffset_y ]
        [ z ]   [ R31 R32 R33 ] [ qfac * pixdim[3] * k ]   [ qoffset_z ]
        
        The qoffset_* shifts are in the NIFTI-1 header.  Note that the center
        of the (i,j,k)=(0,0,0) voxel (first value in the dataset array) is
        just (x,y,z)=(qoffset_x,qoffset_y,qoffset_z).
        
        The rotation matrix R is calculated from the quatern_* parameters.
        This calculation is described below.
        
        The scaling factor qfac is either 1 or -1.  The rotation matrix R
        defined by the quaternion parameters is "proper" (has determinant 1).
        This may not fit the needs of the data; for example, if the image
        grid is
        i increases from Left-to-Right
        j increases from Anterior-to-Posterior
        k increases from Inferior-to-Superior
        Then (i,j,k) is a left-handed triple.  In this example, if qfac=1,
        the R matrix would have to be
        
        [  1   0   0 ]
        [  0  -1   0 ]  which is "improper" (determinant = -1).
        [  0   0   1 ]
        
        If we set qfac=-1, then the R matrix would be

        [  1   0   0 ]
        [  0  -1   0 ]  which is proper.
        [  0   0  -1 ]
        
        This R matrix is represented by quaternion [a,b,c,d] = [0,1,0,0]
        (which encodes a 180 degree rotation about the x-axis).
        
        
        Returns:
        ________________________
        transmatrix :numpy.array
              simple 4X4 transformation matrix
        """
        # check qfac
        qfac = float(self.header['pixdim'][0])
        if qfac not in [-1.0, 1.0]:
            if qfac == 0.0:
                # According to Nifti Spec, if pixdim[0]=0.0, take qfac=1
                print 'qfac of nifti header is invalid: setting to 1.0'
                print 'check your original file to validate orientation'
                qfac = 1.0;
            else:
                raise Nifti1FormatError('invalid qfac: orientation unknown')

        transmatrix = quatern2mat(b=self.header['quatern_b'],
                                  c=self.header['quatern_c'],
                                  d=self.header['quatern_d'],
                                  qx=self.header['qoffset_x'],
                                  qy=self.header['qoffset_y'],
                                  qz=self.header['qoffset_z'],
                                  dx=self.header['pixdim'][1],
                                  dy=self.header['pixdim'][2],
                                  dz=self.header['pixdim'][3],
                                  qfac=qfac)
        return transmatrix

    def _getaffine_method3(self):
        """
        Method to get image orientation location
        based on Method3 in nifti.h

        METHOD 3 (used when sform_code > 0):
        -----------------------------------
        The (x,y,z) coordinates are given by a general affine transformation
        of the (i,j,k) indexes:
        
        x = srow_x[0] * i + srow_x[1] * j + srow_x[2] * k + srow_x[3]
        y = srow_y[0] * i + srow_y[1] * j + srow_y[2] * k + srow_y[3]
        z = srow_z[0] * i + srow_z[1] * j + srow_z[2] * k + srow_z[3]
        
        The srow_* vectors are in the NIFTI_1 header.  Note that no use is
        made of pixdim[] in this method.
        
        Returns:
        ________________________
        transmatrix :numpy.array
              simple 4X4 transformation matrix
        """
        transmatrix = np.zeros((4,4))
        transmatrix[3,3] = 1.0
        
        transmatrix[0] = np.array(self.header['srow_x'])
        transmatrix[1] = np.array(self.header['srow_y'])
        transmatrix[2] = np.array(self.header['srow_z'])
        return transmatrix
        
        
    def _affine_from_header(self):
        """
        Returns appropriate affine transform to send to Sampling Grid 
        to define voxel (array indexes)-> 
             real-world (continuous coordinates) mapping 
        

        Calculates appropriate Affine transform in nipy (c-ordered) data
        (t z y x) needed to generate a SamplingGrid
        Check first for sform > 0
            (gives tranform of image to some standard space)
        Then check for qform > 0
            (gives transform in original scanner space)
        Then do the nifti Method 1 fallback
            (only when qform and sform == 0)

        Returns
        -------
        affine  :  numpy.ndarray (dim[0],dim[0]) 
            ndims X ndims transformation matrix in nipy order 
            | z  0  0  1 |       | t  0  0  0  1 |
            | 0  y  0  1 |   or  | 0  z  0  0  1 |
            | 0  0  x  1 |       | 0  0  y  0  1 |
            | 0  0  0  1 |       | 0  0  0  x  1 |
                                 | 0  0  0  0  1 |

        Examples
        --------
        Can be used to generate a Sampling grid
        from neuroimaging.core.reference import grid
        newgrid = grid.SamplingGrid.from_affine(grid.Affine(affine),
                                               ('zspace', 'yspace', 'xspace'),
                                                tuple(nimg.header['dim'][1:4]))
            
                                 
        """

        if self.header['sform_code'] > 0:
            """
            Method to map into a standard space
              use srow_x,srow_y,srow_z
            """
            value = self._getaffine_method3()

        elif self.header['qform_code'] > 0:
            """
            Method to map into original scanner space
            """
            value = self._getaffine_method2()

        else:
            """
            Using default Method 1
            """
            value = self._getaffine_method1()

        """
        generate transforms to flip data from matlabish
        #  to nipyish ordering
        """
        trans = np.zeros((4,4))
        trans[0:3,0:3] = np.fliplr(np.eye(3))
        trans[3,3] = 1
        trans2 = trans.copy()
        trans2[:,3] = 1
        affine4 = np.dot(np.dot(trans, value), trans2)
        """
        # deal with 4D+ dimensions
        """
        if self.header['dim'][0] > 3:
            # create identity with steps based on pixdim
            affine = np.eye(self.header['dim'][0])
            step = np.array(self.header['pixdim'][1:(self.ndim+1)])
            affine = affine * step[::-1]
            affine[-4:,-4:] = affine4

            return affine
        else:
            return affine4
        
        

        
    def _grid_from_header(self):
        """
        Check first for sform > 0
        (gives tranform of image to some standard space)
        Then check for qform > 0
        (gives transform in scanner space)
        Then do the nifti Method 1 fallback
        (only when qform and sform == 0)
        """
        
            

        origin = (self.header['qoffset_x'],
                  self.header['qoffset_y'],
                  self.header['qoffset_z'])
        origin += (0,) * (self.ndim - 3)
        step = tuple(self.header['pixdim'][1:(self.ndim+1)])
        shape = tuple(self.header['dim'][1:(self.ndim+1)])
        axisnames = ['xspace', 'yspace', 'zspace', 'time', 'vector'][:self.ndim]
        tgrid = SamplingGrid.from_start_step(axisnames,
                                             -np.array(origin),
                                             step,
                                             shape)
        tgridm = tgrid.python2matlab().affine

        # Correct transform information of tgrid by
        # NIFTI file's transform information
        # tgrid's transform matrix may be larger than 4x4,
        # the NIFTI file provides information for the last
        # (in C/python index) three coordinates, or (in
        # MATLAB/FORTRAN index) the first three.

        # What follows is just a way to write over tgrid's
        # transformation matrix with the appropriate information
        # from the NIFTI header

        t = self.transform
        tm = Affine(t).python2matlab().transform
        tgridm[:3,:3] = tm[:3,:3]
        tgridm[:3,-1] = tm[:3,-1]
        self.grid = SamplingGrid(Affine(tgridm).matlab2python(),
                                 tgrid.input_coords,
                                 tgrid.output_coords)

    def _header_from_grid(self):
        """
        Try to set up these fields of the NIFIT1 header from what we know:

        datatype
        bitpix
        quatern_b,c,d
        qoffset_x,y,z
        qfac
        bitpix
        dim

        Note, that for greater than 3d images. The 4x4 matrix
        written to the NIFTI1 file is the submatrix corresponding
        to the three fastest moving coordinates.

        The pixdims of the other dimensions are taken to be the
        corresponding diagonal elements of the transform matrix.

        WARNING:
        --------
        This means that the only way the NIFTI1 file's grid will agree
        with the input grid is if it is diagonal in the dimensions
        other than x,y,z and its origin is 0 because the
        NIFTI1 format does not allow origins for dimensions other than
        x,y,z.
        
        For instance, an fMRI file with this 5x5 transform (in (t,x,y,z) and
        C indexing order) will be fine:

        [[TR  0   0   0  0],
         [0 M11 M12 M13 O1],
         [0 M21 M22 M23 O2],
         [0 M31 M32 M33 O3],
         [0   0   0   0  1]]

        But this one will not
        [[TR  a   b   c  d],
         [e M11 M12 M13 O1],
         [f M21 M22 M23 O2],
         [g M31 M32 M33 O3],
         [0   0   0   0  1]]
        
        if any of a, b, c, d, e, f, g are non zero.
        """

        ndimin, ndimout = self.grid.ndim

        if ndimin != ndimout:
            raise ValueError, 'to create NIFTI1 file, grid should have same number of input and output dimensions'
        self.ndim = ndimin
    
        if not isinstance(self.grid.mapping, Affine):
            raise Nifti1FormatError, 'error: non-Affine grid in writing' \
                  'out NIFTI-1 file'

        ddim = self.ndim - 3
        t = Affine(self.grid.mapping.transform[ddim:,ddim:]).python2matlab().transform

        self.header['datatype'] = sctype2datatype[self.dtype.type]
        self.header['bitpix'] = self.dtype.itemsize * 8

        qb, qc, qd, qx, qy, qz, dx, dy, dz, qfac = mat2quatern(t)

        (self.header['quatern_b'],
         self.header['quatern_c'],
         self.header['quatern_d']) = qb, qc, qd
        
        (self.header['qoffset_x'],
         self.header['qoffset_y'],
         self.header['qoffset_z']) = qx, qy, qz

        self.header['qfac'] = qfac

        _pixdim = [0.]*8
        _pixdim[0:4] = [qfac, dx, dy, dz]
        _pixdim[4:(self.ndim+1)] = np.diag(self.grid.mapping.transform)[0:(self.ndim-3)]
        self.header['pixdim'] = _pixdim

        # this should be set to something, 1 happens
        # to be NIFTI_XFORM_SCANNER_ANAT

        self.header['qform_code'] = 1
        
        self.header['dim'] = \
                        [self.ndim] + list(self.grid.shape) + [0]*(7-self.ndim)
        
    def _transform(self):
        """
        Return 4x4 transform matrix based on the NIFTI attributes
        for the 3d (spatial) part of the mapping.
        If self.sform_code > 0, use the attributes srow_{x,y,z}, else
        if self.qform_code > 0, use the quaternion
        else use a diagonal matrix filled in by pixdim.

        See help(neuroimaging.data_io.formats.nifti1_ext) for explanation.

        It return the 4x4 matrix in NiPy order, rather than MATLAB order.
        That is, entering voxel [0,0,0] gives the (z,y,x) coordinates
        of the first voxel. In MATLAB order, entering voxel [1,1,1] gives
        the (x,y,z) coordinates of the first voxel.
        """

        qfac = float(self.header['pixdim'][0])
        if qfac not in [-1.0, 1.0]:
            if qfac == 0.0:
                # According to Nifti Spec, if pixdim[0]=0.0, take qfac=1
                qfac = 1.0;
            else:
                raise Nifti1FormatError('invalid qfac: orientation unknown')
        
        value = np.zeros((4,4))
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

            value[0] = np.array(self.header['srow_x'])
            value[1] = np.array(self.header['srow_y'])
            value[2] = np.array(self.header['srow_z'])

        return Affine(value).matlab2python().transform 
    transform = property(_transform)
    
    def _getscalers(self):
        if not hasattr(self, "_scalers"):
            def f(y):
                return y*self.header['scl_slope'] + self.header['scl_inter']
            def finv(y):
                return ((y-self.header['scl_inter']) / self.header['scl_slope']).astype(self.dtype)
            self._scalers = [f, finv]
        return self._scalers
    
    scalers = property(_getscalers)
    
    def get_scale_factors(self):
        """Return scale factor information.

        See docstring in BinaryFormat.get_scale_factors
        
        """
        
        return scale_data, self.header['scl_slope'], self.header['scl_inter']

    def postread(self, x):
        """
        NIFTI-1 normalization based on scl_slope and scl_inter.
        """
        print '\t\t Nifti1.postread. BYPASSED postread FUNCTIONALITY!!!'
        return x

        if not self.use_memmap:
            return x

        if self.header['scl_slope']:
            return x * self.header['scl_slope'] + self.header['scl_inter']
        else:
            return x

    def prewrite(self, x):
        """
        NIFTI-1 normalization based on scl_slope and scl_inter.
        If we need to cast the data into Integers, then record the
        new scaling
        """
        # check if a cast is needed in these two cases:
        # 1 - we're replacing all the data
        # 2 - the maximum of the given slice of data exceeds the
        #     global maximum under the current scaling
        #
        # NIFTI1 also contains an intercept term, so see if that needs
        # to change

        if not self.use_memmap:
            return x

        scaled_x = (x - self.header['scl_inter'])/self.header['scl_slope']
        if np.asarray(x).shape == self.data.shape or scaled_x.max() > self.data.max():  
            if x.shape == self.data.shape:
                minval = x.min()
            else:
                minval = min(x.min(), self[:].min())
            # try to find a new intercept if:
            # it's an unsigned type (in order to shift up), or
            # if all values > 0 (in order to shift down)
            if self.dtype in np.sctypes['uint']:
                intercept = minval
            else:
                intercept = minval>0 and minval or 0
                            
            scale = utils.scale_data(x-intercept,
                                     self.dtype, self.header['scl_slope'])

            # if the scale or intercept changed, mark it down
            if scale != self.header['scl_slope'] or \
               intercept != self.header['scl_inter']:
                self.header['scl_inter'] = intercept
                self.header['scl_slope'] = scale
                # be careful with NIFTI, open it rb+ in case we're writing
                # into the same file as the data (.nii file)
                fp = self.datasource.open(self.header_file, 'rb+')

                self.write_header(hdrfile=fp)
                scaled_x = (x - self.header['scl_inter'])/self.header['scl_slope']

        return scaled_x

