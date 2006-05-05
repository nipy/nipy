import types
from path import path
from struct import *
import struct
import enthought.traits as traits
from neuroimaging import import_from
from neuroimaging.attributes import attribute

# struct byte order constants
NATIVE = "="
LITTLE_ENDIAN = "<"
BIG_ENDIAN = ">"

def struct_format(byte_order, elements):
    return byte_order+" ".join(elements)
    
def struct_unpack(infile, byte_order, elements):
    format = struct_format(byte_order, elements)
    return struct.unpack(format, infile.read(struct.calcsize(format)))

def struct_pack(byte_order, elements, values):
    format = struct_format(byte_order, elements)
    return struct.pack(format, *values)


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



##############################################################################
class BinaryHeaderValidator(traits.TraitHandler):

    def __init__(self, packstr, value=None, seek=0, bytesign = '>', **keywords):
        if len(packstr) < 1: raise ValueError("packstr must be nonempty")
        for key, value in keywords.items(): setattr(self, key, value)
        self.seek = seek
        self.packstr = packstr
        self.bytesign = bytesign

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
                       hdrfile.read(calcsize(self.packstr)))
        if not is_tupled(self.packstr, value):
            value = value[0]
        return value

def isseq(value):
    try:
        len(value)
        isseq = True
    except TypeError:
        isseq = False
    return type(value) != types.StringType and isseq

def is_tupled(packstr, value):
    return (packstr[-1] != 's' and len(tuple(value)) > 1)

def BinaryHeaderAtt(packstr, value=None, **keywords):
    validator = BinaryHeaderValidator(packstr, value=value, **keywords)
    return traits.Trait(value, validator)
 
format_modules = (
  "neuroimaging.image.formats.analyze",
  #"neuroimaging.image.formats.afni",
  #"neuroimaging.image.formats.nifti",
  #"neuroimaging.image.formats.minc",
)

#-----------------------------------------------------------------------------
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

#-----------------------------------------------------------------------------
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
