from struct import calcsize, pack, unpack
import sys
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

# what is native and what is swapped?
mybyteorders = {
    sys.byteorder: sys.byteorder=='little' and LITTLE_ENDIAN or BIG_ENDIAN,
    'swapped': sys.byteorder=='little' and BIG_ENDIAN or LITTLE_ENDIAN
    }

# map format chars to python data types
_typemap = dict((
  (("l","L","f","d","q","Q"), float),
  (("h","H","i","I","P"),     int),
  (("x","c","b","B","s","p"), str)))

# All allowed format strings.
allformats = []
for _formats in _typemap.keys():
    allformats.extend(_formats)

format_defaults = {'i': 0, 'I':0,
                   'h': 0, 'H':0,
                   'f': 0.,
                   'c': '\0',
                   'l':0.,
                   's': '',
                   'B': 0}

######## STRUCT PACKING/UNPACKING/INTERPRETATION ROUTINES ####################
def numvalues(format):
    """ The number of values for the given format. """
    numstr, fmtchar = format[:-1], format[-1]
    return (numstr and fmtchar not in ("s","p")) and int(numstr) or 1

def elemtype(format):
    fmtchar = format[-1]
    for formats, typ in _typemap.items():
        if fmtchar in formats:
            return typ
    raise ValueError("format char %s must be one of: %s"%\
                     (fmtchar, allformats))

def sanevalues(format, value):
    nvals, valtype = isinstance(value, (tuple, list)) and \
                     (len(value), type(value[0])) or (1, type(value))
    
    return elemtype(format) == valtype and numvalues(format) == nvals

def formattype(format):
    return numvalues(format) > 1 and list or elemtype(format)

def flatten_values(valseq):
    """ Flattens the type of header values constructed by aggregate. """
    if not isinstance(valseq, list):
        return [valseq]
    if valseq == []:
        return valseq
    return flatten_values(valseq[0]) + flatten_values(valseq[1:])

def takeval(numvals, values):
    """ Take numvals from values.

    Returns a single value if numvals == 1 or else a list of values.
    """
    if numvals == 1:
        return values.pop(0)
    else:
        return [values.pop(0) for _ in range(numvals)]

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

def touch(filename):
    """ Ensure that filename exists. """
    open(filename, 'w')

#### To filter data written to a file
def scale_data(data, new_dtype, default_scale):
    "scales numbers in data to desired match dynamic range of new dtype"
    # if casting to an integer type, check the data range
    # if it clips, then scale down
    # if it has poor integral resolution, then scale up
    if new_dtype in integer_ranges.keys():
        maxval = abs(data.max())
        maxrange = integer_ranges[new_dtype.type]
        scl = maxval/maxrange or 1.
        return scl
    return default_scale
