from types import StringType
from struct import calcsize, pack, unpack
from sys import byteorder
from os.path import exists

from numpy import memmap

from path import path
from attributes import attribute

from neuroimaging import import_from, traits
from neuroimaging.image.utils import writebrick
from neuroimaging.data import iszip, unzip, DataSource
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

    def unpack(infile, byteorder=NATIVE):
        return struct_unpack(infile, byteorder, (self.format,))

    def pack(value, byteorder=NATIVE):
        return struct_pack(byteorder, (self.format,), value)

    def elemtype(self): return elemtype(self.format)
    def formattype(self): return formattype(self.format)

    def set(self, host, value):
        if type(value) is type(""): value = self.fromstring(value)
        attribute.set(self, host, value)

def is_tupled(packstr, value):
    return (packstr[-1] != 's' and len(tuple(value)) > 1)

def isseq(value):
    try:
        len(value)
        isseq = True
    except TypeError:
        isseq = False
    return type(value) != StringType and isseq

class BinaryHeaderValidator(traits.TraitHandler):
    """
    Trait handler used in definition of ANALYZE and NIFTI-1
    headers.
    """
    def __init__(self, packstr, value=None, seek=0, bytesign = '>', **keywords):
        if len(packstr) < 1: raise ValueError("packstr must be nonempty")
        for key, value in keywords.items(): setattr(self, key, value)
        self.seek = seek
        self.packstr = packstr
        self.bytesign = bytesign
        self.size = calcsize(self.packstr)

    def write(self, value, outfile=None):
        if isseq(value): valtup = tuple(value) 
        else: valtup = (value,)
        result = pack(self.bytesign+self.packstr, *valtup)
        if outfile is not None:
            outfile.seek(self.seek)
            outfile.write(result)
        return result

    def validate(self, object, name, value):
        try:
        #if 1:
            result = self.write(value)
        except:
            self.error(object, name, value)

        _value = unpack(self.bytesign + self.packstr, result)
        if is_tupled(self.packstr, _value): return _value
        else: return _value[0]

    def info(self):
        return 'an object of type "%s", apply(struct.pack, "%s", object) must make sense' % (self.packstr, self.packstr)

    def read(self, hdrfile):
        hdrfile.seek(self.seek)
        value = unpack(self.bytesign + self.packstr,
                       hdrfile.read(self.size))
        if not is_tupled(self.packstr, value):
            value = value[0]
        return value

def BinaryHeaderAtt(packstr, seek=0, value=None, **keywords):
    """
    Constructor for a binary header attribute for ANALYZE and NIFTI-1 files.
    """
    validator = BinaryHeaderValidator(packstr, value=value, seek=seek, **keywords)
    return traits.Trait(value, validator)

class BinaryImage(traits.HasTraits):
    """
    A simple way to specify the format of a binary file in terms
    of strings that struct.{pack,unpack} can validate and a byte
    position in the file.

    This is used for both ANALYZE-7.5 and NIFTI-1 files.
    """

    bytesign = traits.Trait(['>','!','<'])
    byteorder = traits.Trait(['little', 'big'])

    # file, mode, datatype

    clobber = traits.false
    filename = traits.Str()
    filebase = traits.Str()
    mode = traits.Trait('r', 'w', 'r+')
    _mode = traits.Trait(['rb', 'wb', 'rb+'])

    # offset for the memmap'ed array

    offset = traits.Int(0)

    # datasource
    
    datasource = traits.Instance(DataSource)

    # grid
    grid = traits.Instance(SamplingGrid)

    def __init__(self, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        if byteorder == 'little': self.bytesign = '<'
        else: self.bytesign = '>'

        self.hdrattnames = [name for name in self.trait_names() \
          if isinstance(self.trait(name).handler, BinaryHeaderValidator)]

    def readheader(self, hdrfile=None):
        self.check_byteorder()
        if hdrfile is None:
            hdrfile = self.datasource.open(self.hdrfilename())
        for traitname in self.hdrattnames:
            trait = self.trait(traitname)
            if hasattr(trait.handler, 'bytesign') and hasattr(self, 'bytesign'):
                trait.handler.bytesign = self.bytesign
            value = trait.handler.read(hdrfile)
            setattr(self, traitname, value)
        self.getdtype()
        hdrfile.close()

    def getdtype(self):
        raise NotImplementedError

    def hdrfilename(self):
        raise NotImplementedError

    def imgfilename(self):
        raise NotImplementedError

    def check_byteorder(self):
        raise NotImplementedError
        
    def writeheader(self, hdrfile=None):

        if hdrfile is None:
            hdrfilename = self.hdrfilename()
            if self.clobber or not exists(self.hdrfilename()):
                hdrfile = file(hdrfilename, 'wb')
            else:
                raise ValueError, 'error writing %s: clobber is False and hdrfile exists' % hdrfilename

        for traitname in self.hdrattnames:
            trait = self.trait(traitname)
            trait.handler.bytesign = self.bytesign

            if hasattr(trait.handler, 'seek'):
                trait.handler.write(getattr(self, traitname), outfile=hdrfile)

        hdrfile.close()

    def getdata(self):
        imgpath = self.imgfilename()
        imgfilename = self.datasource.filename(imgpath)
        if iszip(imgfilename): imgfilename = unzip(imgfilename)
        mode = self.mode in ('r+', 'w') and "r+" or self.mode
        self.memmap = memmap(imgfilename, dtype=self.dtype,
                             shape=tuple(self.grid.shape), mode=mode,
                             offset=self.offset)

    def emptyfile(self):
        """
        Create an empty data file based on
        self.grid and self.dtype
        """
        
        writebrick(file(self.imgfilename(), 'w'),
                   (0,)*self.ndim,
                   N.zeros(self.grid.shape, N.float64),
                   self.grid.shape,
                   byteorder=self.byteorder,
                   outtype=self.dtype)



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
        self.memmap[_slice] = self.prewrite(_data).astype(self.dtype)
        _data.shape = N.product(_data.shape)

    def writeslice(self, _slice, data): self[slice] = data

    def __str__(self):
        value = ''
        for trait in self.hdrattnames:
            value = value + '%s:%s=%s\n' % (self.filebase, trait, str(getattr(self, trait)))
        return value


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

