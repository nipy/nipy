import struct, os, sys, numpy, string, types
from BrainSTAT import Utils
import BrainSTAT.Base.Dimension as Dimension
import BrainSTAT.Base.Coordinates as Coordinates
from BrainSTAT.Base import Mapping
from datatypes import datatypes

_byteorder_dict = {'big':'>', 'little':'<'}

from validators import NIFTI_validators as _validators
from header import NIFTI_header_dict as _header
from header import NIFTI_header_atts as _header_atts

NIFTI_exts = ['.img', '.hdr', '.nii']

dimorder = ['xspace', 'yspace', 'zspace', 'time', 'vector_dimension']

class NIFTIhdr:
    '''A class that implements the nifti1 header with some typechecking. Nifti attributes must conform to their description in nifti1.h.'''
    def __init__(self, niftihdr=None, **keywords):
        # Populate structure with default arguments
        # if a prespecified is specified -- use it
        
        if niftihdr is not None:
            for key, value in _header.items():
                setattr(self, key, getattr(niftihdr, key))
        else:
            for key, value in _header.items():
                setattr(self, key, _header[key][2])

        # Correct any user set values for defaults

        for key, value in keywords.items():
            setattr(self, key, value)

    def __setattr__(self, name, value):
        if name in _header.keys():
            _value = _validators[name](value, bytesign=self.bytesign)
            if _value is not None:
                self.__dict__[name] = _value
        else:
            self.__dict__[name] = value


class NIFTI(NIFTIhdr):
    '''A class to read and write NIFTI format images. You MUST pass the HEADER file as the filename, unlike ANALYZE where either will do. This may be a .hdr file ('ni1' case) or a .nii file ('n+1') case. The code needs to have the header to figure out what kind of file it is.

    >>> from BrainSTAT import *
    >>> from numpy import *
    >>> test = VImage('http://nifti.nimh.nih.gov/nifti-1/data/zstat1.nii.gz', urlstrip='/nifti-1/data/')
    >>> check = VImage(test)
    >>> check.view()
    >>> print check.shape, test.spatial_shape
    (21, 64, 64) (21, 64, 64)

    '''

    reorder_xfm=True
    reorder_dims=True

    def __init__(self, hdrfilename, mode='r', create=False, **keywords):

        ext = os.path.splitext(hdrfilename)[1]
        if ext not in ['.nii', '.hdr']:
            raise ValueError, 'NIFTI files are created with .hdr or .nii files, not .img files.'

        if not os.path.exists(hdrfilename):
            self.clobber = True
        else:
            if keywords.has_key('clobber'):
                self.clobber = keywords['clobber']
            else:
                self.clobber = False

        self.filebase, self.fileext = os.path.splitext(hdrfilename)
        self.hdrfilename = hdrfilename
        
        # figure out machine byte order -- needed for reading binary data

        self.byteorder = sys.byteorder
        self.bytesign = _byteorder_dict[self.byteorder]

        NIFTIhdr.__init__(self, **keywords)

        self.hdrfile = file(self.hdrfilename, mode=mode)

        if mode in ['r+', 'w'] and self.clobber and create:
            dimensions = keyword['dimensions']
            self._dimensions2dim(dimensions)
            self.hdrfile = file(self.hdrfilename, 'wb')
            self.writeheader()
        elif mode in ['r+', 'w'] and self.clobber and not create:
            self.hdrfile = file(self.hdrfilename, 'rb+')
        elif mode not in ['r', 'rb']:
            raise ValueError, 'clobber does not agree with mode'

        self._check_byteorder()
        self.readheader()
        self.typecode = datatypes[self.datatype]

        if self.magic == 'n+1\x00':
            self.brikfile = self.hdrfile
            self.offset = self.vox_offset # should be 352 for most such files
        else:
            if mode in ['r']:
                self.brikfile = file(self.filebase + '.img', mode=mode)
            elif mode in ['r+', 'w'] and self.clobber and create:
                self.brikfile = file(self.filebase + '.img', mode=mode)
            elif mode in ['r+', 'w'] and self.clobber and not create:
                self.brikfile = file(self.filebase + '.img', mode='rb+')
            self.offset = 0

        self.ndim = self.dim[0]
        self.shape = self.dim[1:(1+self.ndim)][::-1]
        self.step = self.pixdim[1:(1+self.ndim)][::-1]
        self.start = [0.] * self.ndim

        ## Setup affine transformation

        self.incoords = Coordinates.VoxelCoordinates('voxel', self.indim)
        self.outcoords = Coordinates.OrthogonalCoordinates('world', self.outdim)

        matrix = self._transform()
        self.mapping = Mapping.Affine(self.incoords, self.outcoords, matrix)
        if NIFTI.reorder_xfm:
            self.mapping.reorder(reorder_dims=NIFTI.reorder_dims)

        self.incoords = self.mapping.input_coords
        self.outcoords = self.mapping.output_coords

        self.start = self.mapping.output_coords.start
        self.step = self.mapping.output_coords.step
        self.shape = self.mapping.output_coords.shape

    def _transform(self):
        """Return 4x4 transform matrix based on the NIFTI attributes."""

        value = numpy.zeros((4,4), numpy.Float)
        value[3,3] = 1.0
        
        if self.sform_code > 0:
            # use 4x4 matrix

            value[0] = self.srow_x
            value[1] = self.srow_y
            value[2] = self.srow_z

        elif self.qform_code > 0:
            # use quaternion matrix
            # this calculation taken from
            # http://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/quatern.html
            
            a, b, c, d = (1.0, self.quatern_b, self.quatern_c, self.quatern_d)
            R = numpy.array([[a*a+b*b-c*c-d*d, 2.*b*c-2*a*d,2*b*d+2*a*c],
                                [2*b*c+2*a*d, a*a+c*c-b*b-d*d, 2*c*d-2*a*b],
                                [2*b*d-2*a*c, 2*c*d+2*a*b, a*a+d*d-c*c-b*b]])
            if self.pixdim[0] == 0.0:
                qfac = 1.0
            else:
                qfac = self.pixdim[0]
            R[:,2] = qfac * R[:,2]

            value[0:3,0:3] = R
            value[0,3] = self.qoffset_x
            value[1,3] = self.qoffset_y
            value[2,3] = self.qoffset_z

        else:
            value[0,0] = self.pixdim[1]
            value[1,1] = self.pixdim[2]
            value[2,2] = self.pixdim[3]

        return value
            
    def _dimensions2dim(self, dimensions):
        '''This routine tries to a list of dimensions into sensible NIFTI dimensions.'''

        _dimnames = [dim.name for dim in dimensions]
        _dimshape = [dim.length for dim in dimensions]
        _dimdict = {}

        for _name in _dimnames:
            _dimdict[_name] = dimensions[_dimnames.index(_name)]
            
        if 'vector_dimension' in _dimnames:
            ndim = 5
            has_vector = True
        else:
            has_vector = False
            if 'time' in _dimnames:
                ndim = 4
                has_time = True
            else:
                has_time = False
                ndim = len(dimensions)

        dim = [ndim]
        pixdim = list(self.pixdim[0:1])

        self.spatial_dimensions = []

        i = 1
        for _name in ['xspace', 'yspace', 'zspace']:
            try: # see if these dimensions exist
                dim.append(_dimdict[_name].length)
                pixdim.append(abs(_dimdict[_name].step))
                self.spatial_dimensions.append(_dimdict[_name])
            except: # else set pixdim=0 even though dimension may be needed
                dim.append(1)
                pixdim.append(0.)
        if has_time and not has_vector:
            dim.append(_dimdict['time'].length)
            pixdim.append(abs(_dimdict['time'].step))
        elif not has_time and has_vector:
            dim.append(1)
            pixdim.append(0.)
            dim.append(_dimdict['vector_dimension'].length)
        elif has_time and has_vector:
            dim.append(_dimdict['time'].length)
            pixdim.append(abs(_dimdict['time'].step))
            dim.append(_dimdict['vector_dimension'].length)

        self.outdim = dimensions
        self.indim = [Dimension.RegularDimension(name=outdim.name, length=outdim.length, start=0.0, step=1.0) for outdim in self.outdim]

        self.dim = tuple(dim + [1] * (8 - len(dim)))
        self.pixdim = tuple(pixdim + [0.] * (8 - len(pixdim)))
        
    def read(self, start, count, **keywords):
        return_value = Utils.brickutils.readbrick(self.brikfile, start, count, self.shape, byteorder=self.byteorder, intype = self.typecode, offset=self.offset)
        if self.scl_slope not in  [1.0, 0.0]:
            return_value = self.scl_slope * return_value
        if self.scl_inter != 0.0:
            return_value = return_value + self.scl_inter
        return return_value

    def write(self, start, data, **keywords):
        self.close()
        self.open(mode='r+', header=False)
        if self.scl_inter != 0:
            outdata = data - self.scl_inter
        else:
            outdata = data
        if self.scl_slope != 1.0:
            outdata = outdata / self.scl_slope
        if len(start) == 3 and len(self.shape) == 4 and self.shape[0] == 1:
            newstart = (0,) + tuple(start) # Is the NIFTI file "really 3d"?
        else:
            newstart = start
        Utils.brickutils.writebrick(self.brikfile, newstart, outdata, self.shape, byteorder = self.byteorder, outtype = self.typecode, offset = self.offset)
        return 

    def close(self, header=True, brick=True):
        if header:
            self.hdrfile.close()
        if brick:
            self.brikfile.close()

    def open(self, mode='r', header=True, brick=True):
        if mode != 'r' and not self.clobber:
            raise ValueError, 'clobber does not agree with mode'
        if mode is None:
            mode = 'r'
        if mode == 'r':
            mode = 'rb'
        if mode == 'w':
            mode = 'wb'
        elif mode == 'r+':
            mode = 'rb+'
        if header:
            try:
                self.hdrfile = file(self.hdrfile.name, mode=mode)
                self.hdrfile.seek(0,0)
            except:
                raise ValueError, 'errors opening header file %s' % self.hdrfile.name
        if brick:
            try:
                self.brikfile = file(self.brikfile.name, mode=mode)
                self.brikfile.seek(0,0)
            except:
                raise ValueError, 'errors opening data file %s' % self.hdrfile.name


    def _check_byteorder(self):
        '''A check of byteorder based on the 'sizeof_hdr' attribute.'''
        self.close(brick=False)
        self.open(mode='r', brick=False)
        try:
            self.hdrfile.seek(0,0)
        except:
            self.hdrfile = file(self.hdrfile.name)
            self.hdrfile.seek(0,0)
        sizeof_hdr = _header_atts[0]
        tmp = self.hdrfile.read(struct.calcsize(self.bytesign + sizeof_hdr[1]))
        if tmp != 348:
            if self.bytesign in ['>', '!']:
                self.bytesign = '<'
                self.byteorder = 'little'
            else:
                self.bytesign = '!'
                self.byteorder = 'big'
        self.close(brick=False)

    def readheader(self):
        self.close(brick=False)
        self.open(mode='r', brick=False)
        try:
            self.hdrfile.seek(0,0)
        except:
            self.hdrfile = file(self.hdrfile.name)
            self.hdrfile.seek(0,0)
        for att in _header_atts:
            tmp = self.hdrfile.read(struct.calcsize(self.bytesign + att[1]))
            value = struct.unpack(self.bytesign + att[1], tmp)
            if len(value) == 1:
                setattr(self, att[0], value[0])
            else:
                setattr(self, att[0], list(value))

        self.close(brick=False)

        dimensions = []
        self.ndim = self.dim[0]
        for i in range(self.ndim):
            if self.pixdim[i+1] != 0:
                dimensions.append(Dimension.RegularDimension(name=dimorder[i], length=self.dim[i+1], start=0.0, step=self.pixdim[i+1]))
        self._dimensions2dim(dimensions)
        return

    def writeheader(self, hdrfile = None):
        if not hdrfile and self.clobber:
            hdrfile = file(self.hdrfile.name, 'w')
        elif self.clobber:
            self.hdrfile.close()
            self.hdrfile = file(self.hdrfile.name, 'w')
            hdrfile = self.hdrfile
        else:
            raise ValueError, 'clobber is False and no hdrfile supplied'
        for att in _header_atts: # Fill in default values if attributes are not present
            if not hasattr(self, att[0]):
                setattr(self, att[0], att[3])
        for att in _atts:
            value = getattr(self, att[0])
            if att[1][-1] == 's':
                value = value.__str__()
            if not att[2]:
                value = (value,)
            hdrfile.write(apply(struct.pack, (self.bytesign + att[1],) + tuple(value)))
        hdrfile.close()

    def __str__(self):
        value = ''
        for att in _header_atts:
            _value = getattr(self, att[0])
            value = value + '%s:%s=%s\n' % (os.path.split(self.hdrfilename)[1], att[0], _value.__str__())
        return value[:-1]


"""
URLPipe class expects this.
"""

creator = NIFTI
valid = NIFTI_exts
