"This module implements details of the Analyze7.5 file format."
import struct

from numpy import memmap, uint8, int16, int32, float32, \
  float64, complex64, amin, amax, dtype, asarray

from attributes import attribute, readonly, deferto, wrapper, scope
from odict import odict
from path import path

from neuroimaging.reference.grid import SamplingGrid
from neuroimaging.reference.axis import space
from neuroimaging.image.formats import struct_unpack, struct_pack, structfield,\
  LITTLE_ENDIAN, BIG_ENDIAN
from neuroimaging.refactoring.baseimage import BaseImage
from neuroimaging.data import DataSource

# datatype is a one bit flag into the datatype identification byte of the
# Analyze header. 
BYTE = 2
SHORT = 4
INTEGER = 8
FLOAT = 16
COMPLEX = 32
DOUBLE = 64 

# map Analyze datatype to Numeric typecode
datatype2typecode = {
  BYTE: uint8,
  SHORT: int16,
  INTEGER: int32,
  FLOAT: float32,
  DOUBLE: float64,
  COMPLEX: complex64}

# map Numeric typecode to Analyze datatype
typecode2datatype = \
  dict([(v,k) for k,v in datatype2typecode.items()])

HEADER_SIZE = 348


##############################################################################
class ElementWrapper (wrapper):
    "Wrap a single element of another sequence attribute."
    classdef=True
    def __init__(self, name, delegate, index, readonly=None):
        if not isinstance(delegate, attribute):
            raise ValueError("delegate must be an attribute")
        doc = "[Wrapper for element %s of %s] "%(index, delegate.name)
        if delegate.__doc__: doc = doc + delegate.__doc__
        attribute.__init__(self, name, doc=doc)
        self.delegate = delegate
        self.index = index
        if readonly is not None: self.readonly = readonly
    def get(self, host):
        delegate = getattr(host,self.delegate.name)
        return delegate[self.index]
    def set(self, host, value):
        if self.readonly:
            raise AttributeError("wrapper %s is read-only"%self.name)
        delegate = getattr(host,self.delegate.name)
        delegate[self.index] = value


##############################################################################
class StructFieldElementWrapper (ElementWrapper):
    classdef=True
    def __init__(self, name, delegate, index, readonly=None):
        if not isinstance(delegate, structfield):
            raise ValueError("delegate must be a structfield")
        ElementWrapper.__init__(self, name, delegate, index, readonly)
        self.implements = (delegate.elemtype(),)


# ordered dictionary of header field names mapped to struct format
struct_fields = odict((
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
    ('origin','3h'),
    ('sunused','4s'),
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

field_formats = struct_fields.values()

#-----------------------------------------------------------------------------
def wrap_elems(delegate, names, indices=()):
    if not indices: indices = range(len(names))
    elif len(names) != len(indices): raise ValueError(
      "names and indices must have the same number of elements")
    scope(1).update(
      dict([(name, ElementWrapper(name,delegate,index))\
            for name,index in zip(names, indices)]))


headeratts = dict(
  [(name, structfield(name,format)) \
  for name,format in struct_fields.items()])


##############################################################################
class AnalyzeHeader (object):
    """
    >>> from neuroimaging.tests.data import repository
    >>> hdr = AnalyzeHeader("rho.hdr", repository.open)
    """
    locals().update(headeratts)

    # convenience interpretations of some dim elements
    wrap_elems(dim, ("ndim","xdim","ydim","zdim","tdim"))

    # convenience interpretations of some pixdim elements
    wrap_elems(pixdim, ("xsize","ysize","zsize","tsize"), (1,2,3,4))

    # convenience interpretation of origin
    wrap_elems(origin, ("x0","y0","z0"))

    class byteorder (attribute): "big or little endian"

    class numpy_dtype (readonly):
        "Numpy datatype object corresponding to the header datatype and byteorder"
        def get(_,self):
            return dtype(
              datatype2typecode[self.datatype]).newbyteorder(self.byteorder)

    class shape (readonly):
        def get(_,self):
            return self.tdim and \
              (self.tdim, self.zdim, self.ydim, self.xdim) or\
              (self.zdim, self.ydim, self.xdim)

    #-------------------------------------------------------------------------
    @staticmethod
    def guess_byteorder(hdrfile):
        """
        Determine byte order of the header.  The first header element is the
        header size.  It should always be 384.  If it is not then you know you
        read it in the wrong byte order.
        """
        if type(hdrfile)==type(""): hdrfile=file(hdrfile)
        byteorder = LITTLE_ENDIAN
        reported_length = struct_unpack(hdrfile,
          byteorder, field_formats[0:1])[0]
        if reported_length != HEADER_SIZE: byteorder = BIG_ENDIAN
        return byteorder

    #-------------------------------------------------------------------------
    def __init__(self, filename, opener=file):

        self.byteorder = AnalyzeHeader.guess_byteorder(opener(filename))

        # unpack all header values
        values = struct_unpack(opener(filename), self.byteorder, field_formats)

        # now load values into self
        map(self.__setattr__, struct_fields.keys(), values)

    #-------------------------------------------------------------------------
    def __str__(self):
        return "\n".join(["%s\t%s"%(att,`getattr(self,att)`) \
           for att in struct_fields.keys()])

    #-------------------------------------------------------------------------
    def write(self, outfile, clobber=False):
        AnalyzeWriter(self, clobber=clobber).write_hdr(outfile)


def get_filestem(filename, extensions):
    filename = path(filename)
    stem, ext = filename.splitext()
    if ext in extensions: return stem
    else: return filename


##############################################################################
class AnalyzeImage (BaseImage):
    """
    >>> from neuroimaging.tests.data import repository
    >>> image = AnalyzeImage("rho", repository.open)
    """
    extensions = (".hdr",".img",".mat")

    class _datasource (readonly): "private datasource"; default=DataSource()
 
    class filestem (readonly):
        "filename minus extensions"
        implements=str
        def set(_,self,value):
            super(readonly,_).set(self, get_filestem(value, self.extensions))
    class hdrfile (readonly):
        "header filename"
        def get(_,self): return self.filestem+".hdr"
    class imgfile (readonly):
        "image filename"
        def get(_,self): return self.filestem+".img"
    class matfile (readonly):
        "matrix filename"
        def get(_,self): return self.filestem+".mat"

    class header (readonly):
        "analyze header"
        implements=AnalyzeHeader
        def init(_, self): return self.load_header()

    # delegate attribute access to header
    deferto(header)

    # overload some header attributes
    class datatype (readonly):
        def get(_,self): return typecode2datatype(self.array.dtype.char)
    class bitpix (readonly):
        def get(_,self): return 8*self.array.dtype.itemsize
    class glmin (readonly): get=lambda _,self: amin(abs(self.array).flat)
    class glmax (readonly): get=lambda _,self: amax(abs(self.array).flat)

    #-------------------------------------------------------------------------
    @staticmethod
    def fromimage(image): pass

    #-------------------------------------------------------------------------
    def __init__(self, filename, datasource=None):
        self.filestem = filename
        if datasource is not None: self._datasource = datasource
        self.header = self.load_header()
        BaseImage.__init__(self, self.load_array(), grid=self.load_grid())

    #-------------------------------------------------------------------------
    def load_header(self):
        return AnalyzeHeader(self.hdrfile, opener=self._datasource.open)

    #-------------------------------------------------------------------------
    def load_array(self):
        arr = memmap(self._datasource.filename(self.imgfile),
            dtype=self.numpy_dtype, shape=self.shape)
        if self.ndim==4 and self.tdim==1: arr=arr.squeeze()
        return arr

    #-------------------------------------------------------------------------
    def load_grid(self):
        """
        Return affine transformation matrix, if it exists.  For now, the format
        is assumed to be a tab-delimited 4 line file.  Other formats should be
        added.
        """
        if self.ndim == 3:
            axisnames = space[::-1]
            origin = self.origin[0:3]
            step = self.pixdim[1:4]
            shape = self.dim[1:4]

        elif self.ndim == 4:
            axisnames = space[::-1] + ('time', )
            origin = tuple(self.origin[0:3]) + (1,)
            step = tuple(self.pixdim[1:5]) 
            shape = self.dim[1:5]
            if self.tdim == 1:
                origin = origin[0:3]
                step = step[0:3]
                axisnames = axisnames[0:3]
                shape = self.dim[1:4]

        ## Setup affine transformation
        self.grid = SamplingGrid.from_start_step(
          names=axisnames, shape=shape, start=-asarray(origin)*step, step=step)

        if self._datasource.exists(self.matfile):
            self.grid.transform(self.readmat())

        # assume .mat matrix uses FORTRAN indexing
        self.grid = self.grid.matlab2python()

    #-------------------------------------------------------------------------
    def write(self, outfile, clobber=False):
        AnalyzeWriter(self, clobber=clobber).write(outfile)



##############################################################################
class AnalyzeWriter (object):
    """
    Write a given image into a single Analyze7.5 format hdr/img pair.
    """
    _field_defaults = {
      'sizeof_hdr': HEADER_SIZE,
      'extents': 16384,
      'regular': 'r',
      'hkey_un0': ' ',
      'vox_units': 'mm',
      'scale_factor':1.}

    _format_defaults = {'i': 0, 'h': 0, 'f': 0., 'c': '\0', 's': ''}

    class _image (readonly): "image being written"
    class _clobber (readonly): "overwrite existing files?"; default=False

    #-------------------------------------------------------------------------
    def __init__(self, image, clobber=None):
        self._image = image
        if clobber is not None: self._clobber = clobber

    #-------------------------------------------------------------------------
    @staticmethod
    def _default_field_value(fieldname, fieldformat):
        "[STATIC] Get the default value for the given field."
        return AnalyzeWriter._field_defaults.get(fieldname, None) or \
               AnalyzeWriter._format_defaults[fieldformat[-1]]

    #-------------------------------------------------------------------------
    def write(self, filestem):
        "Write ANALYZE format header, image file pair."
        headername, imagename = "%s.hdr"%filestem, "%s.img"%filestem
        self.write_hdr(headername)
        self.write_img(imagename)

    #-------------------------------------------------------------------------
    def write_hdr(self, outfile):
        """
        Write ANALYZE format header (.hdr) file.
        @param outfile: filename or filehandle to write the header into
        """
        if type(outfile) == type(""): outfile = file(outfile,'w')

        def fieldvalue(fieldname, fieldformat):
            if hasattr(self._image, fieldname):
                return getattr(self._image, fieldname)
            return AnalyzeWriter._default_field_value(fieldname, fieldformat)

        fieldvalues = [fieldvalue(*field) for field in struct_fields.items()]
        header = struct_pack(self._image.byteorder, field_formats, fieldvalues)
        outfile.write(header)


    #-------------------------------------------------------------------------
    def write_img(self, filename):
        "Write ANALYZE format image (.img) file."
        # Write the image file.
        f = file( filename, "w" )
        f.write( self._image.array.tostring() )
        f.close()

#-----------------------------------------------------------------------------
def _concatenate(listoflists):
    "Flatten a list of lists by one degree."
    finallist = []
    for sublist in listoflists: finallist.extend(sublist)
    return finallist

#-----------------------------------------------------------------------------
def writeImage(image, filestem, datatype=None, targetdim=None):
    """
    Write the given image to the filesystem as one or more Analyze7.5 format
    hdr/img pairs.
    @param filestem:  will be prepended to each hdr and img file.
    @param targetdim:  indicates the dimensionality of data to be written into
      a single hdr/img pair.  For example, if a volumetric time-series is
      given, and targetdim==3, then each volume will get its own file pair.
      Likewise, if targetdim==2, then every slice of each volume will get its
      own pair.
    """
    dimnames = {3:"volume", 2:"slice"}
    def images_and_names(image, stem, targetdim):
        # base case
        if targetdim >= image.ndim: return [(image, stem)]
        
        # recursive case
        subimages = tuple(image.subImages())
        substems = ["%s_%s%d"%(stem, dimnames[image.ndim-1], i)\
                    for i in range(len(subimages))]
        return _concatenate(
          [images_and_names(subimage, substem, targetdim)\
           for subimage,substem in zip(subimages, substems)])

    if targetdim is None: targetdim = image.ndim
    for subimage, substem in images_and_names(image, filestem, targetdim):
        AnalyzeWriter(subimage, datatype=datatype).write(substem)

#-----------------------------------------------------------------------------
def readImage(filename): return AnalyzeImage(filename)


if __name__ == "__main__":
    from doctest import testmod; testmod()
