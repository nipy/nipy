from types import StringType
from struct import calcsize, pack, unpack
from sys import byteorder

from numpy import memmap, zeros, float64

from path import path
from attributes import attribute

from neuroimaging import import_from, traits
from neuroimaging.data import iszip, unzip, DataSource
from neuroimaging.data.header import BinaryHeader, BinaryHeaderAtt
from neuroimaging.reference.grid import SamplingGrid

# struct byte order constants
NATIVE = "="
LITTLE_ENDIAN = "<"
BIG_ENDIAN = ">"

# map format chars to python data types
_typemap = dict((
  (("l","L","f","d","q","Q"), float),
  (("h","H","i","I","P"),     int),
  (("x","c","b","B","s","p"), str)))

# All allowed format strings.
allformats = []
for formats in _typemap.keys(): allformats.extend(formats)

def numvalues(format):
    numstr, fmtchar = format[:-1], format[-1]
    return (numstr and fmtchar not in ("s","p")) and int(numstr) or 1

def elemtype(format):
    fmtchar = format[-1]
    for formats, typ in _typemap.items():
        if fmtchar in formats: return typ
    raise ValueError("format char %s must be one of: %s"%\
                     (fmtchar, allformats()))

def formattype(format):
    return numvalues(format) > 1 and list or elemtype(format)

def takeval(numvalues, values):
    if numvalues==1: return values.pop(0)
    else: return [values.pop(0) for i in range(numvalues)]

def struct_format(byte_order, elements):
    return byte_order+" ".join(elements)
   
def aggregate(formats, values):
    return [takeval(numvalues(format), values) for format in formats]

def struct_unpack(infile, byte_order, elements):
    format = struct_format(byte_order, elements)
    return aggregate(elements,
      list(unpack(format, infile.read(calcsize(format)))))

def struct_pack(byte_order, elements, values):
    format = struct_format(byte_order, elements)
    return pack(format, *values)



class structfield (attribute):
    classdef=True

    def __init__(self, name, format):
        self.format = format
        self.implements = (self.formattype(),)
        attribute.__init__(self, name)
        #if self.default is None: self.default = self._defaults[self.format]

    def fromstring(self, string): return self.formattype()(string)

    def unpack(self, infile, byteorder=NATIVE):
        return struct_unpack(infile, byteorder, (self.format,))

    def pack(self, value, byteorder=NATIVE):
        return struct_pack(byteorder, (self.format,), value)

    def elemtype(self): return elemtype(self.format)
    def formattype(self): return formattype(self.format)

    def set(self, host, value):
        if type(value) is type(""): value = self.fromstring(value)
        attribute.set(self, host, value)

class BinaryImage(BinaryHeader):
    """
    BinaryHeader with a brick of data attached to be memmap'ed.
    """

    # grid

    grid = traits.Instance(SamplingGrid)

    def imgfilename(self):
        raise NotImplementedError

    def getdata(self, offset=0):
        imgpath = self.imgfilename()
        if imgpath == self.hdrfilename():
            offset += self.header_length
        else:
            offset += 0
        imgfilename = self.datasource.filename(imgpath)
        if iszip(imgfilename): imgfilename = unzip(imgfilename)
        mode = self.mode in ('r+', 'w') and "r+" or self.mode
        self.memmap = memmap(imgfilename, dtype=self.dtype,
                             shape=tuple(self.grid.shape), mode=mode,
                             offset=offset)

    def emptyfile(self):
        """
        Create an empty data file based on
        self.grid and self.dtype
        """
        
        outfile = file(self.imgfilename(), 'w')
        outstr = zeros(self.grid.shape[1:], self.dtype).tostring()
        for i in range(self.grid.shape[0]):
            outfile.write(outstr)
        outfile.close()

    def __del__(self):
        if hasattr(self, "memmap"):
            self.memmap.sync()
            del(self.memmap)

    def postread(self, x):
        """
        Point transformation of data post reading.
        """
        return x

    def prewrite(self, x):
        """
        Point transformation of data pre writing.
        """
        return x

    def __getitem__(self, _slice):
        return self.postread(self.memmap[_slice])

    def getslice(self, _slice): return self[_slice]

    def __setitem__(self, _slice, data):
        self.memmap[_slice] = self.prewrite(data).astype(self.dtype)

    def writeslice(self, _slice, data): self[slice] = data



format_modules = (
  "neuroimaging.image.formats.analyze",
  "neuroimaging.image.formats.nifti1",
  #"neuroimaging.image.formats.afni",
  #"neuroimaging.image.formats.minc",
)

def getreader(filename):
    "Return the appropriate image reader for the given file type."
    extension = path(filename).splitext()[1]
    all_extensions = []
    for modname in format_modules:
        reader = import_from(modname, "reader")
        if extension in reader.extensions: return reader
        all_extensions += reader.extensions

    # if we made it this far, a reader was not found
    raise NotImplementedError,\
      "file extension %(ext)s not recognized, %(exts)s files can be created "\
      "at this time."%(extension, all_extensions)


def hasreader(filename):
    """
    Determine if there is an image format reader registered for the given
    file type.
    """
    try:
        getreader(filename)
        return True
    except NotImplementedError:
        return False

