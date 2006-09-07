from numpy.core.memmap import memmap as memmap_type
import numpy as N

from neuroimaging.utils.odict import odict
from neuroimaging.data_io import DataSource
import neuroimaging.sandbox.formats.binary as bin
from neuroimaging.core.reference.axis import space, spacetime
from neuroimaging.core.reference.mapping import Affine
from neuroimaging.core.reference.grid import SamplingGrid
from neuroimaging.utils.path import path

class AnalyzeFormatError(Exception):
    """
    Analyze format error exception
    """


# datatype is a one bit flag into the datatype identification byte of the
# Analyze header. 
BYTE = 2
SHORT = 4
INTEGER = 8
FLOAT = 16
COMPLEX = 32
DOUBLE = 64 

# map Analyze datatype to numpy scalar type
datatype2sctype = {
  BYTE: N.uint8,
  SHORT: N.int16,
  INTEGER: N.int32,
  FLOAT: N.float32,
  DOUBLE: N.float64,
  COMPLEX: N.complex64}

# map numpy scalar type to Analyze datatype
sctype2datatype = \
  dict([(v,k) for k,v in datatype2sctype.items()])

HEADER_SIZE = 348

# ordered dictionary of header field names mapped to struct format
struct_formats = odict((
    ('sizeof_hdr','i'),
    ('data_type','10s'),
    ('db_name','18s'),
    ('extents','i'),
    ('session_error','h'),
    ('regular','c'),
    ('hkey_un0','c'),
    ('dim','8h'),
    ('vox_units','4s'),
    ('cal_units','8s'),
    ('unused1','h'),
    ('datatype','h'),
    ('bitpix','h'),
    ('dim_un0','h'),
    ('pixdim','8f'),
    ('vox_offset','f'),
    ('scale_factor','f'),
    ('funused2','f'),
    ('funused3','f'),
    ('cal_max','f'),
    ('cal_min','f'),
    ('compressed','i'),
    ('verified','i'),
    ('glmax','i'),
    ('glmin','i'),
    ('descrip','80s'),
    ('aux_file','24s'),
    ('orient','c'),
    ('origin','5h'),
    ('generated','10s'),
    ('scannum','10s'),
    ('patient_id','10s'),
    ('exp_date','10s'),
    ('exp_time','10s'),
    ('hist_un0','3s'),
    ('views','i'),
    ('vols_added','i'),
    ('start_field','i'),
    ('field_skip','i'),
    ('omax','i'),
    ('omin','i'),
    ('smax','i'),
    ('smin','i')))

field_formats = struct_formats.values()


class Analyze(bin.BinaryFormat):
    """
    A class to read and write ANALYZE format images. 

    """
    _field_defaults = {
      'sizeof_hdr': HEADER_SIZE,
      'extents': 16384,
      'regular': 'r',
      'hkey_un0': ' ',
      'vox_units': 'mm',
      'scale_factor':1.}
    
    extensions = ('.img', '.hdr', '.mat')
    # maybe I'll implement this when I figure out how
    nvector = -1
    # always false for Analyze
    extendable = False

    def __init__(self, filename, mode="r", datasource=DataSource(), **keywords):
        """
        Constructs a Analyze binary format object with at least a filename
        possible additional keyword arguments:
        grid = Grid object
        sctype = numpy scalar type
        intent = meaning of data
        clobber = allowed to clobber?
        usemat = use mat file?
        """
        bin.BinaryFormat.__init__(self, filename, mode, datasource, **keywords)
        self.clobber = keywords.get('clobber', False)
        self.intent = keywords.get('intent', '')
        self.usematfile = keywords.get('usemat', True)

        self.header_file = self.filebase+".hdr"
        self.data_file = self.filebase+".img"
        self.mat_file = self.filebase+".mat"
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
                raise NotImplementedError("Don't know how to create header info without a grid object")
            self.write_header()
        else:
            self.byteorder = self.guess_byteorder(self.header_file,
                                                  datasource=self.datasource)
            self.read_header()
            self.sctype = datatype2sctype[self.header['datatype']]
            self.ndim = self.header['dim'][0]

        # fill in the canonical list as best we can for Analyze
        self.inform_canonical()

        ########## This could stand a clean-up ################################
        if not self.grid:                
            if self.ndim == 3:
                axisnames = space[::-1]
                origin = tuple(self.header['origin'][0:3])
                step = tuple(self.header['pixdim'][1:4])
                shape = tuple(self.header['dim'][1:4])
            elif self.ndim == 4 and self.nvector <= 1:
                axisnames = spacetime[::-1]
                origin = tuple(self.header['origin'][0:3]) + (1,)
                step = tuple(self.header['pixdim'][1:5]) 
                shape = tuple(self.header['dim'][1:5])

            ## Setup affine transformation        
            self.grid = SamplingGrid.from_start_step(names=axisnames,
                                                shape=shape,
                                                start=-N.array(origin)*step,
                                                step=step)

            if self.usematfile:
                self.grid.transform(self.read_mat())
                # assume .mat matrix uses FORTRAN indexing
                self.grid = self.grid.matlab2python()
        #else: Grid was already assigned by Format constructor
        
        # get memmaped array
        self.attach_data()


    @staticmethod
    def _default_field_value(fieldname, fieldformat):
        "[STATIC] Get the default value for the given field."
        return Analyze._field_defaults.get(fieldname, None) or \
               bin.format_defaults[fieldformat[-1]]
    

    def header_defaults(self):
        for field,format in self.header_formats.items():
            self.header[field] = self._default_field_value(field,format)


    def header_from_given(self):
        self.header['datatype'] = sctype2datatype[self.sctype]
        self.header['bitpix'] = N.dtype(self.sctype).itemsize 
        self.grid = self.grid.python2matlab()
        self.ndim = self.grid.ndim
        
        if not isinstance(self.grid.mapping, Affine):
            raise ValueError, 'error: non-Affine grid in writing out ANALYZE file'

        if self.grid.mapping.isdiagonal():
            _diag = True
        else:
            # what's goin on here??
            _diag = False
            self.write_mat()

        _dim = [0]*8
        _pixdim = [0.] * 8
        _dim[0] = self.ndim
        _dim[1:self.ndim+1] = self.grid.shape[:self.ndim]
        _pixdim[1:self.ndim+1] = _diag and \
                        list(N.diag(self.grid.mapping.transform))[:self.ndim] \
                        or [1.]*self.ndim
        self.header['dim'] = _dim
        self.header['pixdim'] = _pixdim
        if _diag:
            origin = self.grid.mapping.inverse()([0]*self.ndim)
            self.header['origin'] = list(origin) + [0]*(5-origin.shape[0])
        if not _diag:
            self.header['origin'] = [0]*5

        self.grid = self.grid.matlab2python()
                             


    def prewrite(self, x):
        """
        Might transform the data before writing;
        at least confirm sctype
        """
        return x.astype(self.sctype)


    def postread(self, x):
        """
        Might transform the data after getting it from memmap
        """
        return x


    def __del__(self):
        if hasattr(self, 'memmap'):
            if isinstance(self.memmap, memmap_type):
                self.memmap.sync()
            del self.memmap


    def inform_canonical(self, fieldsDict=None):
        if fieldsDict is not None:
            self.canonical_fields = odict(fieldsDict)
        else:
            self.canonical_fields['datasize'] = self.header['bitpix']
            (self.canonical_fields['ndim'],
             self.canonical_fields['xdim'],
             self.canonical_fields['ydim'],
             self.canonical_fields['zdim'],
             self.canonical_fields['tdim']) = self.header['dim'][:5]
            self.canonical_fields['scaling'] = self.header['scale_factor']
        
            

    def read_mat(self):
        """
        Return affine transformation matrix, if it exists.
        For now, the format is assumed to be a tab-delimited 4 line file.
        Other formats should be added.
        """
        if self.datasource.exists(self.mat_file):
            return Affine.fromfile(self.datasource.open(self.mat_file),
                                   delimiter='\t')
        else:
            # FIXME: Do we want to use these?
            if self.ndim == 4:
                names = spacetime[::-1]
            else:
                names = space[::-1]
            return Affine.identity(self.ndim)


    def write_mat(self, matfile=None):
        "Write out the affine transformation matrix."
        if matfile is None:
            matfile = self.mat_file
        if self.clobber or not path(matfile).exists():
            self.grid.mapping.tofile(matfile)


    @staticmethod
    def guess_byteorder(hdrfile, datasource=DataSource()):
        """
        Determine byte order of the header.  The first header element is the
        header size.  It should always be 384.  If it is not then you know you
        read it in the wrong byte order.
        """
        if type(hdrfile)==type(""):
            hdrfile = datasource.open(hdrfile)
        byteorder = bin.LITTLE_ENDIAN
        reported_length = bin.struct_unpack(hdrfile,
          byteorder, field_formats[0:1])[0]
        if reported_length != HEADER_SIZE:
            byteorder = bin.BIG_ENDIAN
        return byteorder

