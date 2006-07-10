from types import StringType
from struct import calcsize, pack, unpack
from sys import byteorder

from path import path
from attributes import attribute

from numpy import sctypes, sctype2char

from neuroimaging import import_from, traits
from neuroimaging.data import DataSource
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

scalar_types = []
for key in ['float', 'complex', 'int', 'uint']:
    scalar_types += [sctype2char(val) for val in sctypes[key]]


class Format(traits.HasTraits):

    """ Valid file name extensions """
    extensions = []

    """ Character representation of scalar (numpy) type """
    scalar_type = traits.Trait(scalar_types)

    """ Clobber, filename, filebase, mode """

    clobber = traits.false
    filename = traits.Str()
    filebase = traits.Str()
    mode = traits.Trait('r', 'w', 'r+')
    bmode = traits.Trait(['rb', 'wb', 'rb+'])

    """ Data source """
    datasource = traits.Instance(DataSource)

    """ Header definition """

    header = traits.List

    """ Sampling grid """

    grid = traits.Trait(SamplingGrid)

    def __init__(self, filename, datasource=DataSource(), **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.datasource = datasource
        self.filename = filename

    @classmethod
    def valid(self, filename, verbose=False, mode='r'):
        """
        Check if filename is valid. If verbose=True, actually try to open
        and read the file.
        """
        try:
            extension = path(filename).splitext()[1]

            if extension not in extension:
                return False
            if verbose:
                # Try to actually instantiate the objects
                self(filename, mode)
        except:
            return False
        return True

    def __str__(self):
        value = ''
        for trait in self.header:
            value = value + '%s:%s=%s\n' % (self.filebase, trait[0], str(getattr(self, trait[0])))
        return value

    def add_header_attribute(self, name, definition, value):
        """
        Add an attribute to the header. Definition should be
        a format string interpretable by struct.pack. 
        """
        raise NotImplementedError

    def set_header_attribute(self, name, attribute, value):
        """
        Set an attribute to the header.
        """
        raise NotImplementedError

    def remove_header_attribute(self, name, attribute):
        """
        Remove an attribute from the header.
        """
        raise NotImplementedError

    def read_header(self, hdrfile=None):
        """
        Read header file.
        """
        raise NotImplementedError

    def write_header(self, hdrfile=None):
        """
        Write header file.
        """
        raise NotImplementedError

    def header_filename(self):
        """
        Get header file name.
        """
        return self.filename

    def __getitem__(self, slice):
        """Data access"""
        raise NotImplementedError

    def __setitem__(self, slice, data):
        """Data access"""
        raise NotImplementedError

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

