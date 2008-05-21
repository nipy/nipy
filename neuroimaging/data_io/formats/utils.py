"""
Utility functions and definitions for the `formats` package.
"""

__docformat__ = 'restructuredtext'

from struct import calcsize, pack, unpack
import sys
import os
import numpy as N


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
_allformats = []
for _formats in _typemap.keys():
    _allformats.extend(_formats)

format_defaults = {'i': 0, 'I':0,
                   'h': 0, 'H':0,
                   'f': 0.,
                   'c': '\0',
                   'l':0.,
                   's': '',
                   'B': 0}

######## STRUCT PACKING/UNPACKING/INTERPRETATION ROUTINES ####################
def numvalues(format):
    """ The number of values for the given format. 
    
    :Parameters:
        `format` : TODO
            TODO
            
    :Returns: int
    """
    numstr, fmtchar = format[:-1], format[-1]
    return (numstr and fmtchar not in ("s","p")) and int(numstr) or 1

def elemtype(format):
    """
    Find the type of a given format string

    :Parameters:
        `format` : string
            A format string

    :Returns:
        ``float`` or ``int`` or ``str``

    :Raises ValueError: if an invalid format character is given.
    """
    fmtchar = format[-1]
    for formats, typ in _typemap.items():
        if fmtchar in formats:
            return typ
    raise ValueError("format char %s must be one of: %s"%\
                     (fmtchar, _allformats))

def sanevalues(format, value):
    """
    :Parameters:
        `format` : TODO
            TODO
        `value` : TODO
            TODO
    
    :Returns: ``bool``
    """
    nvals, valtype = isinstance(value, (tuple, list)) and \
                     (len(value), type(value[0])) or (1, type(value))
    
    return elemtype(format) == valtype and numvalues(format) == nvals

def formattype(format):
    """
    :Parameters:
        `format` : TODO
            TODO

    :Returns: TODO
    """
    return numvalues(format) > 1 and list or elemtype(format)

def flatten_values(valseq):
    """ Flattens the type of header values constructed by aggregate. 
    
    :Parameters:
        `valseq` : TODO
            TODO
    
    :Returns: TODO
    """
    if not isinstance(valseq, list):
        return [valseq]
    if valseq == []:
        return valseq
    return flatten_values(valseq[0]) + flatten_values(valseq[1:])

def takeval(numvals, values):
    """ Take numvals from values.

    :Parameters:
        `numvals` : TODO
            TODO
        `values` : TODO
            TODO

    :Returns: a single value if numvals == 1 or else a list of values.
    """
    if numvals == 1:
        return values.pop(0)
    else:
        return [values.pop(0) for _ in range(numvals)]

def struct_format(byte_order, elements):
    """
    :Parameters:
        `byte_order` : ``string``
            TODO
        `elements` : ``[string]``
            TODO
    
    :Returns: ``string``
    """
    return byte_order+" ".join(elements)
   
def aggregate(formats, values):
    """
    :Parameters:
        `formats` : TODO
            TODO
        `values` : TODO
            TODO
            
    :Returns: TODO
    """
    return [takeval(numvalues(format), values) for format in formats]

def struct_unpack(infile, byte_order, formats):
    """Unpack infile using the format characters in formats.

    Parameters
    ----------
    infile : file_like
        File_like object providing a read method (file, StringIO, Datasource...)
    byte_order : string
        String specifying the byte-order.
        {utils.NATIVE, UTILS.LITTLE_ENDIAN, UTILS.BIG_ENDIAN, '=', '<', '>'}
    formats : list
        List of format charaters as defined by the struct module.
            
    Returns
    -------
    unpacked values : list
        Returns a list of unpacked values from the infile using the format 
        characters in formats.

    Examples
    --------
    >>> import StringIO
    >>> from neuroimaging.data_io.formats import utils
    >>> fp = StringIO.StringIO()
    >>> packed = utils.struct_pack(utils.NATIVE, ['4i'], [1,2,3,4])
    >>> fp.write(packed)
    >>> fp.seek(0)
    >>> unpacked = utils.struct_unpack(fp, utils.NATIVE, ['4i'])
    >>> unpacked
    [[1, 2, 3, 4]]
    >>> fp.close()

    """

    fmt = struct_format(byte_order, formats)
    return aggregate(formats, list(unpack(fmt, infile.read(calcsize(fmt)))))

def struct_pack(byte_order, elements, values):
    """
    :Parameters:
        `byte_order` : string
            The byte order to use. Must be one of NATIVE, BIG_ENDIAN,
            LITLE_ENDIAN
        `elements` : [string]
            A list of format string elements to use
        `value` : [ ... ]
            A list of values to be packed into the format string

    :Returns: ``string``
    """
    format = struct_format(byte_order, elements)
    return pack(format, *flatten_values(values))

def touch(filename):
    """ Ensure that filename exists and is writable.

    :Parameters:
        `filename` : string
            The file to be touched

    :Returns: ``None``
    """

    try:
        open(filename, 'a').close()
    except IOError:
        pass
    os.utime(filename, None)


#### To filter data written to a file

# maximum numbers for different sctype
_integer_ranges = {
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


def scale_data(data, new_dtype, default_scale):
    """ Scales numbers in data to desired match dynamic range of new dtype 
    
    :Parameters:
        `data` : TODO
            TODO
        `new_dtype` : TODO
            TODO
        `default_dtype` : TODO
            TODO
            
    :Returns: TODO
    """
    # if casting to an integer type, check the data range
    # if it clips, then scale down
    # if it has poor integral resolution, then scale up
    if new_dtype in _integer_ranges.keys():
        maxval = abs(data.max())
        maxrange = _integer_ranges[new_dtype.type]
        scl = maxval/maxrange or 1.
        return scl
    return default_scale
