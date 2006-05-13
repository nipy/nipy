"This module implements details of the Analyze7.5 file format."
import struct

from numpy import fromstring, reshape, memmap, UInt8, Int16, Int32, Float32, \
  Float64, Complex32, amin, amax, dtype

from odict import odict
from neuroimaging.image.formats import struct_unpack, struct_pack, NATIVE, \
  LITTLE_ENDIAN, BIG_ENDIAN
from neuroimaging.refactoring.baseimage import BaseImage
from neuroimaging.attributes import attribute, readonly, deferto, wrapper
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
  BYTE: UInt8,
  SHORT: Int16,
  INTEGER: Int32,
  FLOAT: Float32,
  DOUBLE: Float64,
  COMPLEX: Complex32}

# map Numeric typecode to Analyze datatype
typecode2datatype = \
  dict([(v,k) for k,v in datatype2typecode.items()])

HEADER_SIZE = 348

# ordered dictionary of header field names mapped to struct format
struct_fields = odict((
    ('sizeof_hdr','i'),
    ('data_type','10s'),
    ('db_name','18s'),
    ('extents','i'),
    ('session_error','h'),
    ('regular','c'),
    ('hkey_un0','c'),
    ('ndim','h'),
    ('xdim','h'),
    ('ydim','h'),
    ('zdim','h'),
    ('tdim','h'),
    ('dim5','h'),
    ('dim6','h'),
    ('dim7','h'),
    ('vox_units','4s'),
    ('cal_units','8s'),
    ('unused1','h'),
    ('datatype','h'),
    ('bitpix','h'),
    ('dim_un0','h'),
    ('pixdim0','f'),
    ('xsize','f'),
    ('ysize','f'),
    ('zsize','f'),
    ('tsize','f'),
    ('pixdim5','f'),
    ('pixdim6','f'),
    ('pixdim7','f'),
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
    ('x0','h'),
    ('y0','h'),
    ('z0','h'),
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


##############################################################################
class structfield (attribute):
    classdef=True

    _typemap = (
      (("l","L","f","d","q","Q"), float),
      (("h","H","i","I","P"),     int),
      (("x","c","b","B","s","p"), str))

    @staticmethod
    def allformats():
        "All allowed format strings."
        allformats = []
        for formats, typ in structfield._typemap:
            allformats.extend(list(formats))
        return allformats

    def __init__(self, name, format):
        self.format = format
        self.implements = (self.formattype(),)
        attribute.__init__(self, name)
        #if self.default is None: self.default = self._defaults[self.format]

    def fromstring(self, string): return self.formattype()(string)

    def unpack(infile, byteorder=NATIVE):
        return struct_unpack(infile, byteorder, (self.format,))

    def pack(value, byteorder=NATIVE):
        return struct_pack(byteorder, (self.format,), value)

    def formattype(self):
        format = self.format[-1]
        for formats, typ in self._typemap:
            if format in formats: return typ
        raise ValueError("format %s must be one of: %s"%\
                         (format,self.allformats()))

    def set(self, host, value):
        if type(value) is type(""): value = self.fromstring(value)
        attribute.set(self, host, value)


headeratts = dict(
  [(name, structfield(name,format)) \
  for name,format in struct_fields.items()])

ro_headeratts = dict(
  [(name, attribute.clone(att,readonly=True)) \
  for name,att in headeratts.items()])

##############################################################################
class AnalyzeHeader (object):
    """
    >>> from neuroimaging.tests.data import repository
    >>> hdr = AnalyzeHeader("rho.hdr", repository.open)
    """
    locals().update(headeratts)

    class byteorder (attribute): "big or little endian"

    @staticmethod
    def guess_byteorder(hdrfile):
        """
        Determine byte order of the header.  The first header element is the
        header size.  It should always be 384.  If it is not then you know
        you read it in the wrong byte order.
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
    def write(self, outfile): AnalyzeWriter.write_hdr(self, outfile)


##############################################################################
class AnalyzeImage (BaseImage):
    class header (readonly): "the analyze header"; implements=AnalyzeHeader

    # delegate attribute access to header
    deferto(header)

    # overload some header attributes
    class datatype (readonly):
        def get(_,self): return typecode2datatype(self.array.dtype.char)
    class bitpix (readonly):
        def get(_,self): return 8*self.array.dtype.itemsize
    class glmin (readonly): get=lambda _,self: self.amin(abs(self.array).flat)
    class glmax (readonly): get=lambda _,self: self.amax(abs(self.array).flat)

    class numpy_dtype (readonly):
        def get(_,self):
            return dtype(datatype2typecode[self.header.datatype])\
                   .newbyteorder(self.byteorder)
    class shape (readonly):
        def get(_,self): return self.tdim and \
                   (self.tdim, self.zdim, self.ydim, self.xdim)\
                   or (self.zdim, self.ydim, self.xdim)

    #-------------------------------------------------------------------------
    def __init__(self, filestem, datasource=DataSource()):
        self._datasource = datasource
        self.header = AnalyzeHeader(filestem+".hdr", opener=datasource.open)
        arr = self.load_image(filestem+".img")
        BaseImage.__init__(self, arr)

    #-------------------------------------------------------------------------
    def load_image(self, filename):
        print "AnalyzeImage memmap(%s,dtype=%s,shape=%s)"%(self._datasource.filename(filename),
            self.numpy_dtype, self.shape)
        return  memmap(self._datasource.filename(filename),
            dtype=self.numpy_dtype, shape=self.shape)


##############################################################################
class AnalyzeWriter (object):
    """
    Write a given image into a single Analyze7.5 format hdr/img pair.
    """
    _defaults_for_fieldname = {
      'sizeof_hdr': HEADER_SIZE,
      'extents': 16384,
      'regular': 'r',
      'hkey_un0': ' ',
      'vox_units': 'mm',
      'scale_factor':1.}

    _defaults_for_descriptor = {'i': 0, 'h': 0, 'f': 0., 'c': '\0', 's': ''}

    #-------------------------------------------------------------------------
    @staticmethod
    def write_hdr(image, outfile):
        """
        Write ANALYZE format header (.hdr) file.
        @param image: either an AnalyzeHeader or AnalyzeImage
        """
        if type(outfile) == type(""): outfile = file(outfile,'w')

        def fieldvalue(fieldname, fieldformat):
            if hasattr(image, fieldname): return getattr(image, fieldname)
            return AnalyzeWriter._default_field_value(fieldname, fieldformat)

        fieldvalues = [fieldvalue(*field) for field in struct_fields.items()]
        header = struct_pack(image.byteorder, field_formats, fieldvalues)
        outfile.write(header)

    #-------------------------------------------------------------------------
    def _default_field_value(self, fieldname, fieldformat):
        "[STATIC] Get the default value for the given field."
        return self._defaults_for_fieldname.get(fieldname, None) or \
               self._defaults_for_descriptor[fieldformat[-1]]

    #-------------------------------------------------------------------------
    def write(self, filestem):
        "Write ANALYZE format header, image file pair."
        headername, imagename = "%s.hdr"%filestem, "%s.img"%filestem
        self.write_hdr(headername)
        self.write_img(imagename)

    #-------------------------------------------------------------------------
    def write_img(self, filename):
        "Write ANALYZE format image (.img) file."
        # Write the image file.
        f = file( filename, "w" )
        f.write( self.image.data.tostring() )
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
