"This module implements details of the Analyze7.5 file format."
import struct

from numpy import fromstring, reshape, memmap
from pylab import randn, amax, Int8, Int16, Int32, Float32, Float64,\
  Complex32, amin, amax

from odict import odict
from neuroimaging.image.formats import struct_unpack, struct_pack, NATIVE, \
  LITTLE_ENDIAN, BIG_ENDIAN
from neuroimaging.refactoring.baseimage import BaseImage
from neuroimaging.refactoring.attributes import attribute, readonly
from neuroimaging.data import DataSource

# datatype is a bit flag into the datatype identification byte of the Analyze
# header. 
BYTE = 2
SHORT = 4
INTEGER = 8
FLOAT = 16
COMPLEX = 32
DOUBLE = 64

# map Analyze datatype to Numeric typecode
datatype2typecode = {
  BYTE: Int8,
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
    def __init__(self, name, format):
        attribute.__init__(self, name)
        self.format = format

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

    #-------------------------------------------------------------------------
    def __init__(self, filename, opener=file):

        # Determine byte order of the header.  The first header element is the
        # header size.  It should always be 384.  If it is not then you know
        # you read it in the wrong byte order.
        byte_order = LITTLE_ENDIAN
        reported_length = struct_unpack(opener(filename),
          byte_order, field_formats[0:1])[0]
        if reported_length != HEADER_SIZE: byte_order = BIG_ENDIAN

        # unpack all header values
        values = struct_unpack(opener(filename), byte_order, field_formats)

        # now load values into self
        map(self.__setattr__, struct_fields.keys(), values)

    #-------------------------------------------------------------------------
    def __str__(self):
        return "\n".join(["%s = %s"%(att,`getattr(self,att)`) \
           for att in struct_fields.keys()])


##############################################################################
class AnalyzeImage (BaseImage):
    class header (readonly): valtype=AnalyzeHeader
    #deferto(header, readonly=True)

    #-------------------------------------------------------------------------
    def __init__(self, filestem, datasource=DataSource()):
        self._datasource = datasource
        self.header = AnalyzeHeader(filestem+".hdr", opener=datasource.open)
        arr = self.load_image(filestem+".img")
        BaseImage.__init__(self, arr)

    #-------------------------------------------------------------------------
    def __getattr__(self, attname): return getattr(self.header, attname)

    #-------------------------------------------------------------------------
    def load_image(self, filename):
        # bytes per pixel
        #bytepix = self.bitpix/8
        numtype = datatype2typecode[self.datatype]
        dims = self.tdim and (self.tdim, self.zdim, self.ydim, self.xdim)\
                          or (self.zdim, self.ydim, self.xdim)
        #datasize = bytepix * reduce(lambda x,y: x*y, dims)
        #arr = fromstring(self._datasource.open(filename).read(), numtype)
        #return reshape(image, dims)
        return memmap(
          self._datasource.filename(filename), dtype=numtype, shape=dims)


##############################################################################
class AnalyzeWriter (object):
    """
    Write a given image into a single Analyze7.5 format hdr/img pair.
    """

    # map datatype to number of bits per pixel (or voxel)
    datatype2bitpix = {
      BYTE: 8,
      SHORT: 16,
      INTEGER: 32,
      FLOAT: 32,
      DOUBLE: 64,
      COMPLEX: 64}

    #[STATIC]
    _defaults_for_fieldname = {
      'sizeof_hdr': HEADER_SIZE,
      'extents': 16384,
      'regular': 'r',
      'hkey_un0': ' ',
      'vox_units': 'mm',
      'scale_factor':1.}
    #[STATIC]
    _defaults_for_descriptor = {'i': 0, 'h': 0, 'f': 0., 'c': '\0', 's': ''}

    #-------------------------------------------------------------------------
    def __init__(self, image, datatype=None):
        self.image = image
        self.datatype = datatype or typecode2datatype[image.data.typecode()]

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
    def write_hdr(self, filename):
        "Write ANALYZE format header (.hdr) file."
        image = self.image
        data_magnitude = abs(image.data)
        imagevalues = {
          'datatype': self.datatype,
          'bitpix': self.datatype2bitpix[self.datatype],
          'ndim': image.ndim,
          'xdim': image.xdim,
          'ydim': image.ydim,
          'zdim': image.zdim,
          'tdim': image.tdim,
          'xsize': image.xsize,
          'ysize': image.ysize,
          'zsize': image.zsize,
          'tsize': image.tsize,
          'glmin': amin(data_magnitude.flat),
          'glmax': amax(data_magnitude.flat),
          'orient': '\0'}   # kludge alert!  this must be fixed!

        def fieldvalue(fieldname, fieldformat):
            if imagevalues.has_key(fieldname): return imagevalues[fieldname]
            if hasattr(image, fieldname): return getattr(image, fieldname)
            return self._default_field_value(fieldname, fieldformat)

        fieldvalues = [fieldvalue(*field) for field in struct_fields.items()]
        #for f,v in zip(struct_fields.keys(),fieldvalues): print f,"=",`v`
        header = struct_pack(NATIVE, field_formats, fieldvalues)
        file(filename,'w').write(header)

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
