"""
TODO
"""
__docformat__ = 'restructuredtext'

import csv

import numpy as np
import scipy.io as SIO

from neuroimaging.utils.odict import odict
from neuroimaging.data_io.datasource import DataSource
from neuroimaging.data_io.formats import utils, binary
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
  BYTE: np.uint8,
  SHORT: np.int16,
  INTEGER: np.int32,
  FLOAT: np.float32,
  DOUBLE: np.float64,
  COMPLEX: np.complex64}

# map numpy scalar type to Analyze datatype
sctype2datatype = \
  dict([(v, k) for k, v in datatype2sctype.items()])

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


class Analyze(binary.BinaryFormat):
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
    #extensions = ('.img')
    # maybe I'll implement this when I figure out how
    nvector = -1
    # always false for Analyze
    extendable = False

    def __init__(self, filename, mode="r", datasource=DataSource(), use_memmap=True, **keywords):
        """
        Constructs a Analyze binary format object with at least a filename
        possible additional keyword arguments:
         - grid = SamplingGrid object
         - dtype = numpy data type
         - intent = meaning of data
         - clobber = allowed to clobber?
         - usemat = use mat file?
        """
        binary.BinaryFormat.__init__(self, filename, mode, datasource, **keywords)
        self.mat_file = self.filebase + ".mat"
        self.clobber = keywords.get('clobber', False)
        self.intent = keywords.get('intent', '')
        self.usematfile = keywords.get('usemat', True)

        self.header_formats = struct_formats

        # fill the header dictionary in order, with any default values
        self.header_defaults()
        if self.mode[0] is "w":
            # should try to populate the canonical fields and
            # corresponding header fields with info from grid?
            self.byteorder = utils.NATIVE
            self.dtype = np.dtype(keywords.get('dtype', np.float64))
            self.dtype = self.dtype.newbyteorder(self.byteorder)
            self.header['datatype'] = sctype2datatype[self.dtype.type]
            self.header['bitpix'] = self.dtype.itemsize * 8
            if self.grid is not None:
                self._header_from_grid()
            else:
                raise NotImplementedError("Don't know how to create header " \
                                          "info without a grid object")
            self.write_header(clobber=self.clobber)
        else:
            self.byteorder = self.guess_byteorder(self.header_file,
                                                  datasource=self.datasource)
            self.read_header()
            ## make sure we have the correct byteorder 
            tmpsctype = datatype2sctype[self.header['datatype']]
            tmpstr = np.dtype(tmpsctype)
            self.dtype = tmpstr.newbyteorder(self.byteorder)
            self.ndim = self.header['dim'][0]
            self._grid_from_header()

        #else: Grid was already assigned by Format constructor
        
        # get memmaped array
        self.attach_data(use_memmap=use_memmap)


    @staticmethod
    def _default_field_value(fieldname, fieldformat):
        """[STATIC] Get the default value for the given field.
        
        :Parameters:
            `fieldname` : TODO
                TODO
            `fieldformat` : TODO
                TODO
        
        :Returns: TODO
        """
        return Analyze._field_defaults.get(fieldname, None) or \
               utils.format_defaults[fieldformat[-1]]
    

    def header_defaults(self):
        """
        :Returns: ``None``
        """
        for field, format in self.header_formats.items():
            self.header[field] = self._default_field_value(field, format)

    def _grid_from_header(self):

        axisnames = ['xspace', 'yspace', 'zspace', 'time', 'vector'][:self.ndim]
        origin = tuple(self.header['origin'])[:self.ndim]
        shape = tuple(self.header['dim'])[1:(self.ndim+1)]
        step = tuple(self.header['pixdim'])[1:(self.ndim+1)]
        ## Setup affine transformation        
        self.grid = SamplingGrid.from_start_step(axisnames,
                                                 -np.array(origin)*step,
                                                 step,
                                                 shape)
        if self.usematfile:
            t = self.read_mat()
            
            tgrid = SamplingGrid.from_start_step(names=axisnames,
                                                 shape=shape,
                                                 start=-np.array(origin),
                                                 step=step)
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
            
            tm = self.read_mat()
            tgridm[:3,:3] = tm[:3,:3]
            tgridm[:3,-1] = tm[:3,-1]
            self.grid = SamplingGrid(Affine(tgridm).matlab2python(),
                                     tgrid.input_coords,
                                     tgrid.output_coords)


                # assume .mat matrix uses FORTRAN indexing


    def _header_from_grid(self):
        """
        :Returns: ``None``
        """
        self.grid = self.grid.python2matlab()
        ndimin, ndimout = self.grid.ndim
        if ndimin != ndimout:
            raise ValueError, 'to create ANALYZE file, grid should have same number of input and output dimensions'
        self.ndim = ndimin
        
        if not isinstance(self.grid.mapping, Affine):
            raise ValueError, 'non-Affine grid in writing out ANALYZE file'


        dpart = np.diag(np.diag(self.grid.affine))
        _diag = np.allclose(dpart, self.grid.affine)

        if not _diag:
            # Q: what's goin on here??
            #
            # A: we have to store the affine part of the grid information
            #    somehow. if the affine matrix is diagonal, it can be saved
            #    in the ANALYZE format
            #    otherwise, we have to write it out, AND be able to read
            #    it in afterwards on opening the file

            self.write_mat()

        _dim = [0]*8
        _pixdim = [0.] * 8

        _dim = [0]*8
        _pixdim = [0.] * 8

        _dim[0] = self.ndim
        _dim[1:self.ndim+1] = self.grid.shape[:self.ndim]
        if _diag:
            _pixdim[1:self.ndim+1] = \
                         list(np.diag(self.grid.affine))[:self.ndim]
        else:
            _pixdim[1:self.ndim+1] = [1.]*self.ndim

        self.header['dim'] = _dim
        self.header['pixdim'] = _pixdim

        if _diag:
            origin = self.grid.mapping.tovoxel(np.array([0]*self.ndim))
            self.header['origin'] = list(origin) + [0]*(5-origin.shape[0])
        if not _diag:
            self.header['origin'] = [0]*5

        self.grid = self.grid.matlab2python()
                             
    def prewrite(self, x):
        """
        Filter the incoming data. If we're casting to an Integer type,
        record the new scale factor
        
        :Parameters:
            `x` : TODO
                TODO
        
        :Returns: TODO
        """
        
        x = np.asarray(x)
        if not self.use_memmap:
            return x

        # try to cast in two cases:
        # 1 - we're replacing all the data
        # 2 - the maximum of the given slice of data exceeds the
        #     global maximum under the current scaling

        if x.shape == self.data.shape or \
               x.max() > (self.header['scale_factor']*self.data).max():
            #FIXME: I'm not 100% sure that this is correct --timl
            scale = utils.scale_data(x, self.dtype,
                                     self.header['scale_factor'])

            # if the scale changed, mark it down
            if scale != self.header['scale_factor']:
                self.header['scale_factor'] = scale
                self.write_header(clobber=True)

        # this can't be done in place, as we need to change
        # the type of x to a floating point value.
        x = x / self.header['scale_factor']
        if self.dtype in [datatype2sctype[FLOAT],
                          datatype2sctype[DOUBLE],
                          datatype2sctype[COMPLEX]]:
            return x
        else:
            return np.round(x)

    def postread(self, x):
        """
        Might transform the data after getting it from memmap
        
        :Parameters:
            `x` : TODO
                TODO
        
        :Returns` : TODO
        """
        if not self.use_memmap:
             return x

        if self.header['scale_factor'] not in [0,1]:
            return self.scalers[0](x)
        else:
            return x

    def __del__(self):
        """
        :Returns: ``None``
        """
        if hasattr(self, 'data'):
            try:
                self.data.sync()
            except AttributeError:
                pass
            del self.data

    def _getscalers(self):
        if not hasattr("_scalers"):
            def f(y):
                return y * self.header['scale_factor']
            def finv(y):
                return (y / self.header['scale_factor']).astype(self.dtype)
        self._scalers = [f, finv]
        return self._scalers
    
    scalers = property(_getscalers)

    def read_mat(self):
        """
        Return affine transformation matrix, if it exists.
        For now, the format is assumed to be a tab-delimited 4 line file.
        Other formats should be added.

        Read Binary
        import scipy.io as sio
        M = sio.loadmat('my.mat')
        sio.savemat('my_new.mat', M)

        :Returns:
            `Affine`
        """
        if self.datasource.exists(self.mat_file):
            print 'what'*80
            import os
            print os.popen("ls -la %s" % self.mat_file).read()
            print 'now'*80
            mat = SIO.loadmat(self.mat_file)
            # SIO.loadmat puts mat in correct order for a C-ordered array
            # no need to Affine.matlab2python to correct for ordering
            if mat.has_key('mat'):
                return Affine(mat.get('mat'))
            elif mat.has_key('M'):
                return Affine(mat.get('M'))
            else:
                print 'Mat file did not contain Transform!'
                # if all fails give them something?
                return Affine.identity(3)
        else:
            return Affine.identity(3)


    def write_mat(self, matfile=None):
        """Write out the affine transformation matrix.

        :Parameters:
            `matfile` : TODO
                TODO

        :Returns: ``None``
        """
        if matfile is None:
            matfile = self.mat_file
        if self.clobber or not path(matfile).exists():
            mattofile(matfile)

    def _get_filenames(self):
        """
        :Returns: ``string``
        """
        return self.filebase + ".hdr", self.filebase + ".img"


    @staticmethod
    def guess_byteorder(hdrfile, datasource=DataSource()):
        """
        Determine byte order of the header.  The first header element is the
        header size.  It should always be 348.  If it is not then you know you
        read it in the wrong byte order.

        :Parameters:
            `hdrfile` : TODO
                TODO
            `datasource` : `DataSource`
                TODO
                
        :Returns:
            ``string`` : One of utils.LITTLE_ENDIAN or utils.BIG_ENDIAN
        """
        if isinstance(hdrfile, str):
            hdrfile = datasource.open(hdrfile)
        byteorder = utils.LITTLE_ENDIAN
        reported_length = utils.struct_unpack(hdrfile,
          byteorder, field_formats[0:1])[0]
        if reported_length != HEADER_SIZE:
            byteorder = utils.BIG_ENDIAN
        return byteorder

def matfromfile(infile, delimiter="\t"):
    """ Read in an affine transformation matrix from a csv file."""
    if isinstance(infile, str):
        infile = open(infile)
    reader = csv.reader(infile, delimiter=delimiter)
    return np.array(list(reader)).astype(float)

def matfrombin(tstr):
    """
    This is broken -- anyone with mat file experience?
    
    Example
    -------

    >>> SLOW = True
    >>> import urllib
    >>> from neuroimaging.core.reference.mapping import frombin
    >>> mat = urllib.urlopen('http://kff.stanford.edu/nipy/testdata/fiac3_fonc1_0089.mat')
    >>> tstr = mat.read()
    >>> print frombin(tstr)
    [[  2.99893500e+00  -3.14532000e-03  -1.06594400e-01  -9.61109780e+01]
     [ -1.37396100e-02  -2.97339600e+00  -5.31224000e-01   1.20082725e+02]
     [  7.88193000e-02  -3.98643000e-01   3.96313600e+00  -3.32398676e+01]
     [  0.00000000e+00   0.00000000e+00   0.00000000e+00   1.00000000e+00]]
    
    """

    T = np.array(unpack('<16d', tstr[-128:]))
    T.shape = (4, 4)
    return T.T

def matfromstr(tstr, ndim=3, delimiter=None):
    """Read a (ndim+1)x(ndim+1) transform matrix from a string."""
    if tstr.startswith("mat file created by perl"):
        return frombin(tstr) 
    else:
        transform = np.array(tstr.split(delimiter)).astype(float)
        transform.shape = (ndim+1,)*2
        return transform


def matfromxfm(tstr, ndim=3):
    """Read a (ndim+1)x(ndim+1) transform matrix from a string.

    The format being read is that used by the FSL group, for example
    http://kff.stanford.edu/FIAC/fiac0/fonc1/fsl/example_func2highres.xfm
    """
    tstr = tstr.split('\n')
    more = True
    data = []
    outdata = []
    for i in range(len(tstr)):

        if tstr[i].find('/matrix') >= 0:
            for j in range((ndim+1)**2):
                data.append(float(tstr[i+j+1]))

        if tstr[i].find('/outputusermatrix') >= 0:
            for j in range((ndim+1)**2):
                outdata.append(float(tstr[i+j+1]))

    data = np.array(data)
    data.shape = (ndim+1,)*2
    outdata = np.array(outdata)
    outdata.shape = (ndim+1,)*2
    return data, outdata


def matfromurl(turl, ndim=3):
    """
    Read a (ndim+1)x(ndim+1) transform matrix from a URL -- tries to autodetect
    '.mat' and '.xfm'.

    Example
    -------

    >>> SLOW = True
    >>> from numpy import testing
    >>> from neuroimaging.data_io.formats.analyze import matfromurl
    >>> x = matfromurl('http://kff.stanford.edu/nipy/testdata/fiac3_fonc1.txt')
    >>> y = matfromurl('http://kff.stanford.edu/nipy/testdata/fiac3_fonc1_0089.mat')
    >>> testing.assert_almost_equal(x, y, decimal=5)

    """
    urlpipe = urllib.urlopen(turl)
    data = urlpipe.read()
    if turl[-3:] in ['mat', 'txt']:
        return matfromstr(data, ndim=ndim)
    elif turl[-3:] == 'xfm':
        return xfmfromstr(data, ndim=ndim)

def mattofile(self, filename):
    """
    Write the transform matrix to a csv file.
    
    :Parameters:
    filename : ``string``
    The filename to write to

    :Returns: ``None``
    """
    
    matfile = open(filename, 'w')
    writer = csv.writer(matfile, delimiter='\t')
    for row in self.transform: 
        writer.writerow(row)
    matfile.close()

