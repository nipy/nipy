from types import TupleType, ListType
from numpy import memmap
from struct import calcsize, pack, unpack
from sys import byteorder
import os

from neuroimaging.utils.odict import odict
#from neuroimaging.data_io.formats import Format
from neuroimaging.data_io import DataSource
from neuroimaging.sandbox.refactoring.formats import Format

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

format_defaults = {'i': 0, 'h': 0, 'f': 0., 'c': '\0', 's': ''}

def numvalues(format):
    numstr, fmtchar = format[:-1], format[-1]
    return (numstr and fmtchar not in ("s","p")) and int(numstr) or 1

def elemtype(format):
    fmtchar = format[-1]
    for formats, typ in _typemap.items():
        if fmtchar in formats: return typ
    raise ValueError("format char %s must be one of: %s"%\
                     (fmtchar, allformats()))

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

##############################################################################
class BinaryFormat(Format):

    # In case they are different files
    header_file = ""
    data_file = ""
    filebase = ""

    byteorder = NATIVE
    
    # Has metadata + possibly extended data,
    # but ALSO has struct formats

    header_formats = odict()
    ext_header_formats = odict()

    extendable = False

    #-------------------------------------------------------------------------
    def __init__(self, filename, mode="r", datasource=DataSource(), **keywords):
        # keep BinaryFormats dealing with datasource and filename/mode
        Format.__init__(self, datasource, keywords.get('grid', None))
        self.mode = mode
        self.filebase = os.path.splitext(filename)[0]
        
        
    #-------------------------------------------------------------------------
    def read_header(self):
        # Populate header dictionary from a file
        values = struct_unpack(open(self.header_file), self.byteorder,
                               self.header_formats.values())
        
        for field, val in zip(self.header.keys(), values):
            self.header[field] = val

    #-------------------------------------------------------------------------
    def write_header(self):
        fp = open(self.header_file, 'wb')
        packed = struct_pack(self.byteorder, self.header_formats.values(),
                             self.header.values())
        fp.write(packed)
        if self.extendable and self.ext_header != {}:
            packed_ext = struct_pack(self.byteorder,
                                     self.ext_header_formats.values(),
                                     self.ext_header.values())
            fp.write(packed_ext)
        fp.close()
    
    #-------------------------------------------------------------------------
    def attach_data(self, offset=0):
        mode = self.mode in ('r+','w') and 'r+' or self.mode
        if mode == 'r+' and not os.path.exists(self.data_file):
            touch(self.data_file)
        self.memmap = memmap(self.datasource.filename(self.data_file),
                             dtype=self.sctype, shape=tuple(self.grid.shape),
                             mode=mode, offset=offset)

    #-------------------------------------------------------------------------
##     def empty_datafile(self, offset=0):
##         if not exists(self.data_file):
##             fp = open(self.data_file, 'wb')
##         else:
##             fp = open(self.data_file, 'rb+')

##         # need to see if we're clobbering header file here,
##         # i'm just writing for now

##         outfile.seek(offset)
        
    #-------------------------------------------------------------------------
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

    #-------------------------------------------------------------------------
    def remove_header_field(self, field):
        if field in self.ext_header.keys():
            self.ext_header.pop(field)
            self.ext_header_formats.pop(field)


    #------------------------------------------------------------------------- 
    #broken!
    def set_header_field(self, field, value):
        try:
            self.header[field] = sanevalues(self.header_formats[field], value) and value
        except KeyError:
            try:
                self.ext_header[field] = \
                        sanevalues(self.ext_header_formats[field], value) and value
            except KeyError:
                raise KeyError
    
    #------------------------------------------------------------------------- 
    def get_header_field(self, field):
        try:
            self.header[field]
        except KeyError:
            try:
                self.ext_header[field]
            except KeyError:
                raise KeyError
    
            
##############################################################################
            
