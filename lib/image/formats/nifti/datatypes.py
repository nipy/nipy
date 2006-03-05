from constants import *
import numpy

datatypes = {DT_NONE:None, # fail if unknown
             DT_UNKNOWN:None, 
             DT_BINARY:numpy.Bool,
             DT_UNSIGNED_CHAR:numpy.UInt8,
             DT_SIGNED_SHORT:numpy.Int16,
             DT_SIGNED_INT:numpy.Int32,
             DT_FLOAT:numpy.Float32,
             DT_COMPLEX:numpy.Complex32,
             DT_DOUBLE:numpy.Float64,
             DT_RGB:None,
             DT_ALL:None,
             DT_UINT8:numpy.UInt8,
             DT_INT16:numpy.Int16,
             DT_INT32:numpy.Int32,
             DT_FLOAT32:numpy.Float32,
             DT_COMPLEX64:numpy.Complex64,
             DT_FLOAT64:numpy.Float64,
             DT_RGB24:None,
             DT_INT8:numpy.Int8,
             DT_UINT16:numpy.UInt16,
             DT_UINT32:numpy.UInt32,
             DT_INT64:numpy.Int64,
             DT_UINT64:numpy.UInt64,
             DT_FLOAT128:None,
             DT_COMPLEX128:None,
             DT_COMPLEX256:None}
