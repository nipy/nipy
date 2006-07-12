import os, sys
from struct import unpack

import numpy as N
from path import path

from neuroimaging.data import iszip, unzip, DataSource
from neuroimaging.reference.axis import space, spacetime
from neuroimaging.reference.mapping import Affine, Mapping
from neuroimaging.reference.grid import SamplingGrid
from neuroimaging.image.utils import writebrick

from neuroimaging.image.formats import BinaryHeaderAtt, BinaryImage, traits

class ANALYZE(BinaryImage):
    """
    A class to read and write ANALYZE format images. 

    """
    # header fields
    sizeof_hdr = BinaryHeaderAtt('i', seek=0, value=348)
    data_type = BinaryHeaderAtt('10s', seek=4, value=' '*10)
    db_name = BinaryHeaderAtt('18s', seek=14, value=' '*18)
    extents = BinaryHeaderAtt('i', seek=32, value=0)
    session_error = BinaryHeaderAtt('h', seek=36, value=0)
    regular = BinaryHeaderAtt('s', seek=38, value='r')
    hkey_un0 = BinaryHeaderAtt('s', seek=39, value='0')
    dim = BinaryHeaderAtt('8h', seek=40, value=(4,91,109,91,1,0,0,0))
    vox_units = BinaryHeaderAtt('4s', seek=56, value='mm  ')
    calib_units = BinaryHeaderAtt('8s', seek=60, value=' '*8)
    unused1 = BinaryHeaderAtt('h', seek=68, value=0)
    datatype = BinaryHeaderAtt('h', seek=70, value=16)
    bitpix = BinaryHeaderAtt('h', seek=72, value=8)
    dim_un0 = BinaryHeaderAtt('h', seek=74, value=0)
    pixdim = BinaryHeaderAtt('8f', seek=76, value=(0.,2.,2.,2.,)+(0.,)*4)
    vox_offset = BinaryHeaderAtt('f', seek=108, value=0.)
    scale_factor = BinaryHeaderAtt('f', seek=112, value=1.)
    funused2 = BinaryHeaderAtt('f', seek=116, value=0.)
    funused3 = BinaryHeaderAtt('f', seek=120, value=0.)
    calmax = BinaryHeaderAtt('f', seek=124, value=0.)
    calmin = BinaryHeaderAtt('f', seek=128, value=0.)
    compressed = BinaryHeaderAtt('i', seek=132, value=0)
    verified = BinaryHeaderAtt('i', seek=136, value=0)
    glmax = BinaryHeaderAtt('i', seek=140, value=0)
    glmin = BinaryHeaderAtt('i', seek=144, value=0)
    descrip = BinaryHeaderAtt('80s', seek=148, value=' '*80)
    auxfile = BinaryHeaderAtt('24s', seek=228, value='none' + ' '*20)
    orient = BinaryHeaderAtt('B', seek=252, value=0)
    origin = BinaryHeaderAtt('5H', seek=253, value=(46,64,37,0,0))
    generated = BinaryHeaderAtt('10s', seek=263, value=' '*10)
    scannum = BinaryHeaderAtt('10s', seek=273, value=' '*10)
    patient_id = BinaryHeaderAtt('10s', seek=283, value=' '*10)
    exp_date = BinaryHeaderAtt('10s', seek=293, value=' '*10)
    exp_time = BinaryHeaderAtt('10s', seek=303, value=' '*10)
    hist_un0 = BinaryHeaderAtt('3s', seek=313, value=' '*3)
    views = BinaryHeaderAtt('i', seek=316, value=0)
    vols_added = BinaryHeaderAtt('i', seek=320, value=0)
    start_field = BinaryHeaderAtt('i', seek=324, value=0)
    field_skip = BinaryHeaderAtt('i', seek=328, value=0)
    omax = BinaryHeaderAtt('i', seek=332, value=0)
    omin = BinaryHeaderAtt('i', seek=336, value=0)
    smax = BinaryHeaderAtt('i', seek=340, value=0)
    smin = BinaryHeaderAtt('i', seek=344, value=0)

    # file extensions recognized by this format
    extensions = ('.img', '.hdr', '.mat')

    # Use mat file if it's there?
    # This will cause a problem for 4d files occasionally
    usematfile = traits.true

    # Try to squeeze 3d files?
    squeeze = traits.true

    # Vector axis?
    nvector = traits.Int(-1)

    def __init__(self, filename=None, datasource=DataSource(), grid=None, **keywords):

        BinaryImage.__init__(self, **keywords)
        self.datasource = datasource
        self.filebase = filename and os.path.splitext(filename)[0] or None

        if self.mode is 'w':
            self._dimfromgrid(grid)
            self.writeheader()
            if filename: self.readheader()
            self.ndim = len(grid.shape)
            self.emptyfile()
            
        elif filename:
            self.readheader()
            self.ndim = self.dim[0]

        self.customize()

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
        
        self.grid = SamplingGrid.from_start_step(names=axisnames,
                                        shape=shape,
                                        start=-N.array(origin)*step,
                                        step=step)

        if self.usematfile: self.grid.transform(self.readmat())

        # assume .mat matrix uses FORTRAN indexing
        self.grid = self.grid.matlab2python()

        # get memmaped array
        self.getdata()

    def customize(self):
        """
        Customization of the ANALYZE reader, i.e. for ANALYZE_FSL.
        This is done before constructing the grid, and
        can be used to set the header arguments to specific values.
        
        """
        return 


    def _datatype_changed(self):
        self.getdtype()

    def _byteorder_changed(self):
        self.bytesign = {'big':'>', 'little':'<'}[self.byteorder]

    def _bytesign_changed(self):
        self.byteorder = {'>':'big', '<':'little'}[self.bytesign]

    def hdrfilename(self):
        return '%s.hdr' % self.filebase

    def imgfilename(self):
        return '%s.img' % self.filebase

    def matfilename(self):
        return '%s.mat' % self.filebase

    def _grid_changed(self):
        try:
            self.ndim = len(self.grid.shape)
        except:
            pass

    def getdtype(self):
        ANALYZE_Byte = 2
        ANALYZE_Short = 4
        ANALYZE_Int = 8
        ANALYZE_Float = 16
        ANALYZE_Double = 64

        datatypes = {
            ANALYZE_Byte:N.uint8,
            ANALYZE_Short:N.int16,
            ANALYZE_Int:N.int32,
            ANALYZE_Float:N.float32,
            ANALYZE_Double:N.float64}

        self.dtype = N.dtype(datatypes[self.datatype])
        self.dtype = self.dtype.newbyteorder(self.bytesign)

    def postread(self, x):
        """
        ANALYZE normalization based on scale_factor.
        """
        if self.scale_factor:
            return x * self.scale_factor
        else:
            return x

    def prewrite(self, x):
        """
        ANALYZE normalization based on scale_factor.
        """
        if self.scale_factor:
            return x / self.scale_factor
        else:
            return x
        
    def _mode_changed(self):
        _modemap = {'r':'rb', 'w':'wb', 'r+': 'rb+'}
        self._mode = _modemap[self.mode]
        

    def _dimfromgrid(self, grid):
        self.grid = grid.python2matlab()
            
        if not isinstance(self.grid.mapping, Affine):
            raise ValueError, 'error: non-Affine grid in writing out ANALYZE file'

        if self.grid.mapping.isdiagonal():
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
        
    def readmat(self):
        """
        Return affine transformation matrix, if it exists.
        For now, the format is assumed to be a tab-delimited 4 line file.
        Other formats should be added.
        """
        if self.datasource.exists(self.matfilename()):
            return Mapping.fromfile(self.datasource.open(self.matfilename()),
                     input='world', output='world', delimiter='\t')
        else:
            if self.ndim == 4: names = spacetime[::-1]
            else: names = space[::-1]
            return Mapping.identity(
              self.ndim, input='world', output='world', names=names)

    def writemat(self, matfile=None):
        "Write out the affine transformation matrix."
        if matfile is None: matfile = self.matfilename()
        if self.clobber or not path(matfile).exists():
            self.grid.mapping.tofile(matfile)

    def check_byteorder(self):
        """
        Try to guess big/little endianness of an ANALYZE file based on dim[0].
        """

        hdrfile = self.datasource.open(self.hdrfilename())
        for order, sign in {'big':'>', 'little':'<'}.items():
            hdrfile.seek(40)
            x = hdrfile.read(2)
            try:
                test = unpack(sign + 'h', x)[0]
                if test in range(1,8):
                    byteorder, bytesign = order, sign
            except:
                pass
        try:
            self.byteorder, self.bytesign = byteorder, bytesign
        except:
            raise ValueError, 'file format not recognized: byteorder check failed'
        hdrfile.close()

class ANALYZE_FSL(ANALYZE):

    
    def customize(self):
        """
        Customizations to ANALYZE reader that tries to follow
        FSL's use of the ANALYZE-7.5 header.
        """
        self.origin = [1]*5
        self.pixdim = [N.fabs(pixd) for pixd in self.pixdim]



# plug in as a format creator (see formats.getreader)
reader = ANALYZE
