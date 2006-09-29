from struct import calcsize, pack, unpack
from types import TupleType, ListType

import numpy as N


# maximum numbers for different sctype
integer_ranges = {
  N.int8:  127.,
  N.uint8: 255.,
  N.int16: 32767.,
  N.uint16: 65535.,  
  # pretty sure these are better than single floating point precision:
  #N.int32: 2147483647.,
  #N.uint32: 4294967295,,
  #N.int64: 9223372036854775807.,
  #N.uint: 18446744073709551615.,
  }


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

format_defaults = {'i': 0,'I':0, 'h': 0, 'H':0, 'f': 0., 'c': '\0',\
                   'l':0.,'s': '', 'B': 0}

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

def touch(fname):
    open(fname, 'w')

#### To filter data written to a file
def cast_data(data, new_sctype, default_scale):
    "casts numbers in data to desired typecode in data_code"
    # if casting to an integer type, check the data range
    # if it clips, then scale down
    # if it has poor integral resolution, then scale up
    if new_sctype in integer_ranges.keys():
        maxval = abs(data.max())
        maxrange = integer_ranges[new_sctype]
        scl = maxval/maxrange or 1.
        return scl, N.round(data/scl)
    return default_scale, data
