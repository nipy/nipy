import struct, os, sys, numpy, string, types
import numpy as N
from neuroimaging.image import utils
from neuroimaging.reference.axis import VoxelAxis, RegularAxis, space, spacetime
from neuroimaging.reference.coordinate_system import VoxelCoordinateSystem, DiagonalCoordinateSystem
from neuroimaging.reference.mapping import Affine, IdentityMapping
import neuroimaging.reference.mapping as mapping
from neuroimaging.reference.grid import SamplingGrid, fromStartStepLength, python2matlab, matlab2python
from neuroimaging.image.formats.validators import BinaryHeaderAtt, BinaryHeaderValidator
import enthought.traits as traits

_byteorder_dict = {'big':'>', 'little':'<'}

ANALYZE_Byte = 2
ANALYZE_Short = 4
ANALYZE_Int = 8
ANALYZE_Float = 16
ANALYZE_Double = 64

datatypes = {ANALYZE_Byte:(numpy.UInt8, 1),
             ANALYZE_Short:(numpy.UInt16, 2),
             ANALYZE_Int:(numpy.Int32, 4),
             ANALYZE_Float:(numpy.Float32, 4),
             ANALYZE_Double:(numpy.Float64, 8)
             }

ANALYZE_exts = ['.img', '.hdr', '.mat']

class ANALYZEHeaderValidator(BinaryHeaderValidator):
    pass

def ANALYZEHeaderAtt(packstr, value=None, **keywords):
    validator = ANALYZEHeaderValidator(packstr, value=value, **keywords)
    return traits.Trait(value, validator)

class ANALYZEhdr(traits.HasTraits):

    # SPM header definition

    sizeof_hdr = ANALYZEHeaderAtt('i', seek=0, value=348)
    data_type = ANALYZEHeaderAtt('10s', seek=4, value=' '*10)
    db_name = ANALYZEHeaderAtt('18s', seek=14, value=' '*18)
    extents = ANALYZEHeaderAtt('i', seek=32, value=0)
    session_error = ANALYZEHeaderAtt('h', seek=36, value=0)
    regular = ANALYZEHeaderAtt('s', seek=38, value='r')
    hkey_un0 = ANALYZEHeaderAtt('s', seek=39, value='0')
    dim = ANALYZEHeaderAtt('8h', seek=40, value=(4,91,109,91,1,0,0,0))
    vox_units = ANALYZEHeaderAtt('4s', seek=56, value='mm  ')
    calib_units = ANALYZEHeaderAtt('8s', seek=60, value=' '*8)
    unused1 = ANALYZEHeaderAtt('h', seek=68, value=0)
    datatype = ANALYZEHeaderAtt('h', seek=70, value=16)
    bitpix = ANALYZEHeaderAtt('h', seek=72, value=8)
    dim_un0 = ANALYZEHeaderAtt('h', seek=74, value=0)
    pixdim = ANALYZEHeaderAtt('8f', seek=76, value=(0.,2.,2.,2.,)+(0.,)*4)
    vox_offset = ANALYZEHeaderAtt('f', seek=108, value=0.)
    funused1 = ANALYZEHeaderAtt('f', seek=112, value=1.)
    funused2 = ANALYZEHeaderAtt('f', seek=116, value=0.)
    funused3 = ANALYZEHeaderAtt('f', seek=120, value=0.)
    calmax = ANALYZEHeaderAtt('f', seek=124, value=0.)
    calmin = ANALYZEHeaderAtt('f', seek=128, value=0.)
    compressed = ANALYZEHeaderAtt('i', seek=132, value=0)
    verified = ANALYZEHeaderAtt('i', seek=136, value=0)
    glmax = ANALYZEHeaderAtt('i', seek=140, value=0)
    glmin = ANALYZEHeaderAtt('i', seek=144, value=0)
    descrip = ANALYZEHeaderAtt('80s', seek=148, value=' '*80)
    auxfile = ANALYZEHeaderAtt('24s', seek=228, value='none' + ' '*20)
    orient = ANALYZEHeaderAtt('B', seek=252, value=0)
    origin = ANALYZEHeaderAtt('5H', seek=253, value=(46,64,37,0,0))
    generated = ANALYZEHeaderAtt('10s', seek=263, value=' '*10)
    scannum = ANALYZEHeaderAtt('10s', seek=273, value=' '*10)
    patient_id = ANALYZEHeaderAtt('10s', seek=283, value=' '*10)
    exp_date = ANALYZEHeaderAtt('10s', seek=293, value=' '*10)
    exp_time = ANALYZEHeaderAtt('10s', seek=303, value=' '*10)
    hist_un0 = ANALYZEHeaderAtt('3s', seek=313, value=' '*3)
    views = ANALYZEHeaderAtt('i', seek=316, value=0)
    vols_added = ANALYZEHeaderAtt('i', seek=320, value=0)
    start_field = ANALYZEHeaderAtt('i', seek=324, value=0)
    field_skip = ANALYZEHeaderAtt('i', seek=328, value=0)
    omax = ANALYZEHeaderAtt('i', seek=332, value=0)
    omin = ANALYZEHeaderAtt('i', seek=336, value=0)
    smax = ANALYZEHeaderAtt('i', seek=340, value=0)
    smin = ANALYZEHeaderAtt('i', seek=344, value=0)

    filebase = traits.Str()
    clobber = traits.false

    byteorder = traits.Trait(['big', 'little'])
    bytesign = traits.Trait(['>', '<'])

    def _byteorder_changed(self):
        self.bytesign = {'big':'>', 'little':'<'}[self.byteorder]

    def _bytesign_changed(self):
        self.byteorder = {'>':'big', '<':'little'}[self.bytesign]

    def _filebase_changed(self):
        try:
            hdrfile = file(self.hdrfilename())
            if self.mode in ['r', 'r+']:
                self.byteorder, self.bytesign = guess_endianness(hdrfile)
            else:
                self.byteorder = sys.byteorder 
            hdrfile.close()
        except:
            pass
    
    def __init__(self, filename=None, **keywords):

        if filename is not None:
            self.filebase = os.path.splitext(filename)[0]

        traits.HasTraits.__init__(self ,**keywords)
        self.hdrattnames = []
        for name in self.trait_names():
            trait = self.trait(name)
            if isinstance(trait.handler, ANALYZEHeaderValidator):
                self.hdrattnames.append(name)

        if self.filebase:
            if self.mode in ['r', 'r+']:
                self.readheader()

    def __str__(self):
        value = ''
        for trait in self.hdrattnames:
            value = value + '%s:%s=%s\n' % (self.filebase, trait, str(getattr(self, trait)))
        return value

    def readheader(self, hdrfile=None):

        if hdrfile is None:
            hdrfile = file(self.hdrfilename(), 'rb')

        for traitname in self.hdrattnames:
            trait = self.trait(traitname)
            if hasattr(trait.handler, 'bytesign') and hasattr(self, 'bytesign'):
                trait.handler.bytesign = self.bytesign
            value = trait.handler.read(hdrfile)
            setattr(self, traitname, value)

        self.typecode, self.byte = datatypes[self.datatype]

        hdrfile.close()

        return

    def hdrfilename(self):
        return '%s.hdr' % self.filebase

    def imgfilename(self):
        return '%s.img' % self.filebase

    def matfilename(self):
        return '%s.mat' % self.filebase

    def _datatype_changed(self):
        ## TODO / WARNING, datatype is not checked very carefully...

        if self.datatype == ANALYZE_Byte:
            self.bitpix = 8
            self.glmin = 0
            self.glmax = 255
            self.funused1 = abs(self.calmin) / 255
        elif self.datatype == ANALYZE_Short: 
            self.bitpix = 16
            self.funused1 = max(abs(self.calmin), abs(self.calmax)) / (2.0**15-1)
            self.glmin = round(self.funused1 * self.calmin)
            self.glmax = round(self.funused1 * self.calmax)
        elif self.datatype == ANALYZE_Int: 
            self.bitpix = 32
            self.funused1 = max(abs(self.calmin), abs(self.calmax)) / (2.0**31-1)
            self.glmin = round(self.funused1 * self.calmin)
            self.glmax = round(self.funused1 * self.calmax)
        elif self.datatype == ANALYZE_Float:
            self.bitpix = 32
            self.funused1 = 1
            self.glmin = 0
            self.glmax = 0
        elif self.datatype == ANALYZE_Double:
            self.bitpix = 64
            self.funused1 = 1
            self.glmin = 0
            self.glmax = 0
        else:
            raise ValueError, 'invalid datatype'

    def writeheader(self, hdrfile=None):

        if hdrfile is None:
            hdrfilename = self.hdrfilename()
            if self.clobber or not os.path.exists(self.hdrfilename()):
                hdrfile = file(hdrfilename, 'wb')
            else:
                raise ValueError, 'error writing %s: clobber is False and hdrfile exists' % hdrfilename

        for traitname in self.hdrattnames:
            trait = self.trait(traitname)
            trait.handler.bytesign = self.bytesign

            if hasattr(trait.handler, 'seek'):
                trait.handler.write(getattr(self, traitname), outfile=hdrfile)

        hdrfile.close()

class ANALYZE(ANALYZEhdr):

    """
    A class to read and write ANALYZE format images. 

    >>> from BrainSTAT import *
    >>> from numpy import *
    >>> test = VImage(testfile('test.img'))
    >>> check = VImage(test)
    >>> print int(add.reduce(check.readall().flat))
    -11996
    >>> print check.shape, test.shape
    (68, 95, 79) (68, 95, 79)

    >>> test = VImage('http://nifti.nimh.nih.gov/nifti-1/data/zstat1.nii.gz')
    >>> new = test.tofile('test.img')
    >>> print new.shape
    (21, 64, 64)
    >>> print new.ndim
    3
    >>> new.view()

    """

    # file, mode, datatype
    
    memmapped = traits.true
    filename = traits.Str()
    mode = traits.Trait('r', 'w', 'r+')
    _mode = traits.Trait(['rb', 'wb', 'rb+'])
    clobber = traits.false

    # Use mat file if it's there?
    # This will cause a problem for 4d files occasionally

    usematfile = traits.true

    # Ignore the origin as FSL does
    # This is __EQUIVALENT_TO__ setting origin=(1,)*5
    
    ignore_origin = traits.false

    # Use abs(pixdim)?
    
    abs_pixdim = traits.false

    # Try to squeeze 3d files?

    squeeze = traits.true

    # Vector axis?

    nvector = traits.Int(-1)

    # grid

    grid = traits.Any()

    def _grid_changed(self):
        try:
            self.ndim = len(self.grid.shape)
        except:
            pass

    def _datatype_changed(self):
        self.getdtype()
        
    def getdtype(self):
        self.typecode, self.byte = datatypes[self.datatype]
        self.dtype = N.dtype(self.typecode)
        self.dtype = self.dtype.newbyteorder(self.byteorder)

    def _mode_changed(self):
        _modemap = {'r':'rb', 'w':'wb', 'r+': 'rb+'}
        self._mode = _modemap[self.mode]
        
    def _dimfromgrid(self, grid):
        self.grid = python2matlab(grid)
            
        if not isinstance(self.grid.mapping, Affine):
            raise ValueError, 'error: non-Affine grid in writing out ANALYZE file'

        if mapping.isdiagonal(self.grid.mapping.transform[0:self.ndim,0:self.ndim]):
            _diag = True
        else:
            _diag = False
            self.writemat()

        _dim = [0]*8
        _pixdim = [0.] * 8
        _dim[0] = self.ndim

        for i in range(self.ndim):
            _dim[i+1] = self.grid.shape[i]
            if _diag:
                _pixdim[i+1] = self.grid.mapping.transform[i,i]
            else:
                _pixdim[i+1] = 1.
        self.dim = _dim
        self.pixdim = _pixdim
        if _diag:
            origin = self.grid.mapping.map([0]*self.ndim, inverse=True)
            self.origin = list(origin) + [0]*(5-origin.shape[0])
        if not _diag:
            self.origin = [0]*5
        
    def __init__(self, **keywords):

        ANALYZEhdr.__init__(self, **keywords)
        traits.HasTraits.__init__(self, **keywords)

        if self.mode is 'w':
            self._dimfromgrid(keywords['grid'])
            self.writeheader()
            self.getdtype()

            # create empty file

            utils.writebrick(file(self.imgfilename(), 'w'),
                             (0,)*self.ndim,
                             N.zeros(self.grid.shape, N.Float),
                             self.grid.shape,
                             byteorder=self.byteorder,
                             outtype = self.typecode)

        self.readheader()

        self.ndim = self.dim[0]
        
        if self.ignore_origin:
            self.origin = [1]*5

        if self.abs_pixdim:
            self.pixdim = [N.fabs(pixd) for pixd in self.pixdim]

        if self.ndim == 3:
            axisnames = space[::-1]
            origin = self.origin[0:3]
            step = self.pixdim[1:4]
            shape = self.dim[1:4]
        elif self.ndim == 4 and self.nvector <= 1:
            axisnames = space[::-1] + ['time']
            origin = tuple(self.origin[0:3]) + (1,)
            step = tuple(self.pixdim[1:5]) 
            shape = self.dim[1:5]
            if self.squeeze:
                if self.dim[4] == 1:
                    origin = origin[0:3]
                    step = step[0:3]
                    axisnames = axisnames[0:3]
                    shape = self.dim[1:4]
        elif self.ndim == 4 and self.nvector > 1:
            axisnames = ['vector_dimension'] + space[::-1]
            origin = (1,) + self.origin[0:3]
            step = (1,) + tuple(self.pixdim[1:4])  
            shape = self.dim[1:5]
            if self.squeeze:
                if self.dim[1] == 1:
                    origin = origin[1:4]
                    step = step[1:4]
                    axisnames = axisnames[1:4]
                    shape = self.dim[2:5]

        ## Setup affine transformation
                
        self.grid = fromStartStepLength(names=axisnames,
                                        shape=shape,
                                        step=step,
                                        start=-N.array(origin)*step)

        if self.usematfile:
            mat = self.readmat()
            self.grid.mapping = mat * self.grid.mapping

        self.grid = matlab2python(self.grid) # assumes .mat
                                             # matrix is FORTRAN indexing

        if self.memmapped:
            if self.mode is 'r':
                self.memmap = N.memmap(self.imgfilename(), dtype=self.dtype,
                                       shape=tuple(self.grid.shape), mode='r')
            elif self.mode in ['r+', 'w']:
                self.memmap = N.memmap(self.imgfilename(), dtype=self.dtype,
                                       shape=tuple(self.grid.shape), mode='r+')

    def __del__(self):

        if self.memmapped:
            self.memmap.sync()
        del(self.memmap)
        
    def getslice(self, slice):
        v = self.memmap[slice]
        if self.funused1:
            return v * self.funused1
        else:
            return v

    def writeslice(self, slice, data):
        if self.funused1:
            _data = data / self.funused1
        else:
            _data = data
        self.memmap[slice] = _data.astype(self.dtype)
        _data.shape = N.product(_data.shape)
        
    def readmat(self):
        """
        Return affine transformation matrix, if it exists.
        For now, the format is assumed to be a tab-delimited 4 line file.
        Other formats should be added.
        """

        if os.path.exists(self.matfilename()):
            m = mapping.fromfile(self.matfilename(),
                                 input='world',
                                 output='world',
                                 delimiter='\t')
            return m
        else:
            if self.ndim == 4:
                names = spacetime[::-1]
            else:
                names = space[::-1]
            return IdentityMapping(self.ndim, input='world', output='world', names=names)

    def writemat(self, matfile=None):
        """
        Write out the affine transformation matrix.
        """

        if matfile is None:
            matfile = self.matfilename()

        if self.clobber or not os.path.exists(matfile):
            mapping.tofile(self.grid.mapping, matfile)

def guess_endianness(hdrfile):
    """
    Try to guess big/little endianness of an ANALYZE file based on dim[0].
    """

    for order, sign in {'big':'>', 'little':'<', 'net':'!'}.items():
        hdrfile.seek(40)
        x = hdrfile.read(2)
        try:
            test = struct.unpack(sign + 'h', x)[0]
            if test in range(1,8):
                return order, sign
        except:
            raise ValueError 
    raise ValueError, 'file format not recognized: endianness test failed'


"""
URLPipe class expects this.
"""

creator = ANALYZE
valid = ANALYZE_exts
