from types import TupleType, ListType

from numpy.core import memmap as memmap_type
from numpy import memmap
import numpy as N
from struct import calcsize, pack, unpack
import os

from neuroimaging.utils.odict import odict
from neuroimaging.sandbox.formats import Format
from neuroimaging.data_io import DataSource
#from neuroimaging.sandbox.refactoring.formats import Format

class BinaryFormatError(Exception):
    """
    Binary format error exception
    """

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

format_defaults = {'i': 0, 'h': 0, 'f': 0., 'c': '\0', 's': '', 'B': 0}

######## STRUCT PACKING/UNPACKING/INTERPRETATION ROUTINES ####################
def numvalues(format):
    numstr, fmtchar = format[:-1], format[-1]
    return (numstr and fmtchar not in ("s","p")) and int(numstr) or 1

def elemtype(format):
    fmtchar = format[-1]
    for formats, typ in _typemap.items():
        if fmtchar in formats: return typ
    raise ValueError("format char %s must be one of: %s"%\
                     (fmtchar, allformats))

def sanevalues(format, value):
    nvals, valtype = type(value) in (TupleType, ListType) and \
                     (len(value), type(value[0])) or (1, type(value))
    
    return elemtype(format) == valtype and numvalues(format) == nvals

def formattype(format):
    return numvalues(format) > 1 and list or elemtype(format)

def flatten_values(valseq):
    # flattens the type of header values constructed by aggregate
    if type(valseq) != type([]): return [valseq]
    if valseq == []: return valseq
    return flatten_values(valseq[0]) + flatten_values(valseq[1:])

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
    return pack(format, *flatten_values(values))

def touch(fname): open(fname, 'w')


class BinaryFormat(Format):


    # Subclass objects should define these:
    #header_file = ""
    #data_file = ""
    #byteorder = NATIVE
    #extendable = False


    def __init__(self, filename, mode="r", datasource=DataSource(), **keywords):
        # keep BinaryFormats dealing with datasource and filename/mode
        Format.__init__(self, datasource, keywords.get('grid', None))        
        self.mode = mode
        self.filename = filename
        self.filebase = os.path.splitext(filename)[0]
        self.header_formats = odict()
        self.ext_header_formats = odict()
        self.clobber = keywords.get('clobber', False)

        if self.clobber and 'w' in self.mode:
            try:
                os.remove(self.datasource.filename(self.data_file))
            except:
                pass


    def read_header(self):
        # Populate header dictionary from a file
        values = struct_unpack(self.datasource.open(self.header_file),
                               self.byteorder,
                               self.header_formats.values())
        for field, val in zip(self.header.keys(), values):
            self.header[field] = val


    def write_header(self,hdrfile=None):
        # If someone wants to write a headerfile somewhere specific,
        # handle that case immediately
        # Otherwise, try to write to the object's header file

        if hdrfile:
            fp = type(hdrfile) == type('') and open(hdrfile,'wb') or hdrfile
        elif self.datasource.exists(self.header_file):
            fp = self.datasource.open(self.header_file, 'wb')
        else:
            fp = open(self.header_file, 'wb')
        packed = struct_pack(self.byteorder,
                             self.header_formats.values(),
                             self.header.values())
        fp.write(packed)

        if self.extendable and self.ext_header != {}:
            packed_ext = struct_pack(self.byteorder,
                                     self.ext_header_formats.values(),
                                     self.ext_header.values())
            fp.write(packed_ext)

        # close it if we opened it
        if not hdrfile or type(hdrfile) is not type(fp):
            fp.close()


    def attach_data(self, offset=0):

        mode = self.mode in ('r+','w','wb') and 'readwrite' or 'readonly'
        if mode == 'readwrite' and not self.datasource.exists(self.data_file):
            touch(self.data_file)
        self.data = memmap(self.datasource.filename(self.data_file),
                             dtype=self.sctype, shape=tuple(self.grid.shape),
                             mode=mode, offset=offset)


    def prewrite(self, x):
        raise NotImplementedError


    def postread(self, x):
        raise NotImplementedError

    def __getitem__(self, slicer):
        return N.asarray(self.postread(self.data[slicer].newbyteorder(self.byteorder)))


    def __setitem__(self, slicer, data):
        if self.data._mode not in ('r+','w+','w'):
            print "Warning: memapped array is not writeable!"
            return
        self.data[slicer] = \
            self.prewrite(data).astype(self.sctype).newbyteorder(self.byteorder)

    def __del__(self):
        if hasattr(self, 'memmap'):
            if isinstance(self.data, memmap_type):
                self.data.sync()
            del(self.data)

    #### These methods are extraneous, the header dictionaries are
    #### unprotected and can be looked at directly
    def add_header_field(self, field, format, value):
        if not self.extendable:
            raise NotImplementedError("%s header type not "\
                                      "extendable"%self.filetype)
        if field in (header.keys() + ext_header.keys()):
            raise ValueError("Field %s already exists in "
                             "the current header"%field)
        if not sanevalues(format, value):
            raise ValueError("Field format and value(s) do not match up.")

        # if we got here, we're good
        self.ext_header_formats[field] = format
        self.ext_header[field] = value


    def remove_header_field(self, field):
        if field in self.ext_header.keys():
            self.ext_header.pop(field)
            self.ext_header_formats.pop(field)



    def set_header_field(self, field, value):
        try:
            if sanevalues(self.header_formats[field], value):
                self.header[field] = value
        except KeyError:
            try:
                if sanevalues(self.ext_header_formats[field], value):
                    self.ext_header[field] = value
            except KeyError:
                raise KeyError('Field does not exist')
    

    def get_header_field(self, field):
        try:
            return self.header[field]
        except KeyError:
            try:
                return self.ext_header[field]
            except KeyError:
                raise KeyError
    
            

            
