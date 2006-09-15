import os
from struct import unpack

import numpy as N

from neuroimaging import traits

from neuroimaging.data_io import DataSource
from neuroimaging.data_io.formats.binary import BinaryFormat
from neuroimaging.core.reference.axis import space, spacetime
from neuroimaging.core.reference.mapping import Affine
from neuroimaging.core.reference.grid import SamplingGrid
from neuroimaging.utils.path import path

ANALYZE_Byte = 2
ANALYZE_Short = 4
ANALYZE_Int = 8
ANALYZE_Float = 16
ANALYZE_Double = 64

datatypes = {N.uint8:ANALYZE_Byte,
             N.int16:ANALYZE_Short,
             N.int32:ANALYZE_Int,
             N.float32:ANALYZE_Float,
             N.float64:ANALYZE_Double}

sctypes = {
    ANALYZE_Byte:N.uint8,
    ANALYZE_Short:N.int16,
    ANALYZE_Int:N.int32,
    ANALYZE_Float:N.float32,
    ANALYZE_Double:N.float64}


header = [('sizeof_hdr', 'i', 348),
          ('data_type', '10s', ' '*10),
          ('db_name', '18s', ' '*18),
          ('extents', 'i', 0),
          ('session_error', 'h', 0),
          ('regular', 's', 'r'),
          ('hkey_un0', 's', '0'),
          ('dim', '8h', (4,91,109,91,1,0,0,0),),
          ('vox_units', '4s', 'mm  '),
          ('calib_units', '8s', ' '*8),
          ('unused1', 'h', 0),
          ('datatype', 'h', 16),
          ('bitpix', 'h', 8),
          ('dim_un0', 'h', 0),
          ('pixdim', '8f', (0.,2.,2.,2.,)+(0.,)*4),
          ('vox_offset', 'f', 0.),
          ('scale_factor', 'f', 1.),
          ('funused2', 'f', 0.),
          ('funused3', 'f', 0.),
          ('calmax', 'f', 0.),
          ('calmin', 'f', 0.),
          ('compressed', 'i', 0),
          ('verified', 'i', 0),
          ('glmax', 'i', 0),
          ('glmin', 'i', 0),
          ('descrip', '80s', ' '*80),
          ('auxfile', '24s', 'none' + ' '*20),
          ('orient', 'B', 0),
          ('origin', '5H', (46,64,37,0,0)),
          ('generated', '10s', ' '*10),
          ('scannum', '10s', ' '*10),
          ('patient_id', '10s', ' '*10),
          ('exp_date', '10s', ' '*10),
          ('exp_time', '10s', ' '*10),
          ('hist_un0', '3s', ' '*3),
          ('views', 'i', 0),
          ('vols_added', 'i', 0),
          ('start_field', 'i', 0),
          ('field_skip', 'i', 0),
          ('omax', 'i', 0),
          ('omin', 'i', 0),
          ('smax', 'i', 0),
          ('smin', 'i', 0)
          ]

class ANALYZE(BinaryFormat):
    """
    A class to read and write ANALYZE format images. 

    """

    header = traits.ReadOnly(header)


    # file extensions recognized by this format
    extensions = ('.img', '.hdr', '.mat')

    # Use mat file if it's there?
    # This will cause a problem for 4d files occasionally
    usematfile = traits.true

    # Try to squeeze 3d files?
    squeeze = traits.true

    # Vector axis?
    nvector = traits.Int(-1)

    def __init__(self, filename=None, datasource=DataSource(), grid=None,
                 sctype=N.float64, **keywords):

        BinaryFormat.__init__(self, filename, **keywords)

        self.datasource = datasource
        self.filebase = filename and os.path.splitext(filename)[0] or None

        if self.mode is 'w':
            self.sctype = sctype
            self._dimfromgrid(grid)
            self.write_header()
            if filename: self.read_header()
            self.ndim = len(grid.shape)
            self.emptyfile()
            
        elif filename:
            self.read_header()
            self.ndim = self.dim[0]

        self.customize()

        print self.pixdim, self.dim, self.datatype, self.sctype

        if self.ndim == 3:
            axisnames = space[::-1]
            origin = self.origin[0:3]
            step = self.pixdim[1:4]
            shape = self.dim[1:4]
        elif self.ndim == 4 and self.nvector <= 1:
            axisnames = space[::-1] + ('time', )
            origin = tuple(self.origin[0:3]) + (1,)
            step = tuple(self.pixdim[1:5]) 
            shape = self.dim[1:5]
            if self.squeeze and self.dim[4] == 1:
                origin = origin[0:3]
                step = step[0:3]
                axisnames = axisnames[0:3]
                shape = self.dim[1:4]
        elif self.ndim == 4 and self.nvector > 1:
            axisnames = ('vector_dimension', ) + space[::-1]
            origin = (1,) + self.origin[0:3]
            step = (1,) + tuple(self.pixdim[1:4])  
            shape = self.dim[1:5]

            if self.squeeze and self.dim[1] == 1:
                origin = origin[1:4]
                step = step[1:4]
                axisnames = axisnames[1:4]
                shape = self.dim[2:5]

        ## Setup affine transformation
        print "shape = ", shape
        self.grid = SamplingGrid.from_start_step(names=axisnames,
                                        shape=shape,
                                        start=-N.array(origin)*step,
                                        step=step)

        if self.usematfile: self.grid.transform(self.read_mat())

        # assume .mat matrix uses FORTRAN indexing
        self.grid = self.grid.matlab2python()

        # get memmaped array
        self.attach_data()

    def customize(self):
        """
        Customization of the ANALYZE reader, i.e. for ANALYZE_FSL.
        This is done before constructing the grid, and
        can be used to set the header arguments to specific values.
        
        """
        return 


    def _datatype_changed(self):
        self.get_dtype()

    def _byteorder_changed(self):
        self.bytesign = {'big':'>', 'little':'<'}[self.byteorder]

    def _bytesign_changed(self):
        self.byteorder = {'>':'big', '<':'little'}[self.bytesign]

    def header_filename(self):
        return '%s.hdr' % self.filebase

    def image_filename(self):
        return '%s.img' % self.filebase

    def mat_filename(self):
        return '%s.mat' % self.filebase

    def _grid_changed(self):
        try:
            self.ndim = len(self.grid.shape)
        except:
            pass

    def _sctype_changed(self, sctype):

        self.datatype = datatypes[sctype]

    def _datatype_changed(self, datatype):

        self.sctype = sctypes[self.datatype]

    def get_dtype(self):
        self.dtype = N.dtype(self.sctype)
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
            self.write_mat()

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
            origin = self.grid.mapping.map.inverse()([0]*self.ndim)
            self.origin = list(origin) + [0]*(5-origin.shape[0])
        if not _diag:
            self.origin = [0]*5
       
    def read_mat(self):
        """
        Return affine transformation matrix, if it exists.
        For now, the format is assumed to be a tab-delimited 4 line file.
        Other formats should be added.
        """
        if self.datasource.exists(self.mat_filename()):
            return Affine.fromfile(self.datasource.open(self.mat_filename()),
                     input='world', output='world', delimiter='\t')
        else:
            if self.ndim == 4: names = spacetime[::-1]
            else: names = space[::-1]
            return Affine.identity(self.ndim)

    def write_mat(self, matfile=None):
        "Write out the affine transformation matrix."
        if matfile is None: matfile = self.mat_filename()
        if self.clobber or not path(matfile).exists():
            self.grid.mapping.tofile(matfile)

    def check_byteorder(self):
        """
        Try to guess big/little endianness of an ANALYZE file based on dim[0].
        """

        hdrfile = self.datasource.open(self.header_filename())
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

## ANALYZE.headeratts = []
## ANALYZE.headerlength = 0

## for headeratt in headeratts:
##     add_headeratt(ANALYZE, headeratt)

class ANALYZE_FSL(ANALYZE):

    
    def customize(self):
        """
        Customizations to ANALYZE reader that tries to follow
        FSL's use of the ANALYZE-7.5 header.
        """
        self.origin = [1]*5
        self.pixdim = [N.fabs(pixd) for pixd in self.pixdim]


