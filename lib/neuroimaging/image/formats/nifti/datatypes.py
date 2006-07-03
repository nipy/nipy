import constants as C
import numpy as N

datatypes = {C.DT_NONE:None, # fail if unknown
             C.DT_UNKNOWN:None, 
             C.DT_BINARY:N.bool8,
             C.DT_UNSIGNED_CHAR:N.uint8,
             C.DT_SIGNED_SHORT:N.int16,
             C.DT_SIGNED_INT:N.int32,
             C.DT_FLOAT:N.float32,
             C.DT_COMPLEX:None,
             C.DT_DOUBLE:N.float64,
             C.DT_RGB:None,
             C.DT_ALL:None,
             C.DT_UINT8:N.uint8,
             C.DT_INT16:N.int16,
             C.DT_INT32:N.int32,
             C.DT_FLOAT32:N.float32,
             C.DT_COMPLEX64:N.complex64,
             C.DT_FLOAT64:N.float64,
             C.DT_RGB24:None,
             C.DT_INT8:N.int8,
             C.DT_UINT16:N.uint16,
             C.DT_UINT32:N.uint32,
             C.DT_INT64:N.int64,
             C.DT_UINT64:N.uint64,
             C.DT_FLOAT128:None,
             C.DT_COMPLEX128:None,
             C.DT_COMPLEX256:None}
