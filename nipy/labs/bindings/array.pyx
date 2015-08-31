# -*- Mode: Python -*-  Not really, but the syntax is close enough

"""
Python access to core fff functions written in C. This module is
mainly used for unitary tests.

Author: Alexis Roche, 2008.
"""

__version__ = '0.1'

# Includes
from fff cimport *

# Initialize numpy
fffpy_import_array()
import_array()
import numpy as np

# Binded routines
def array_get(A, size_t x, size_t y=0, size_t z=0, size_t t=0):
    """
    Get array element.
    va = array_get(A, size_t x, size_t y=0, size_t z=0, size_t t=0):
    """
    cdef fff_array* a
    cdef double va
    a = fff_array_fromPyArray(A)
    va = fff_array_get(a, x, y, z, t)
    fff_array_delete(a)
    return va


def array_get_block( A, size_t x0, size_t x1, size_t fX=1,
                     size_t y0=0, size_t y1=0, size_t fY=1,
                     size_t z0=0, size_t z1=0, size_t fZ=1,
                     size_t t0=0, size_t t1=0, size_t fT=1 ):
    """
    Get block
    Asub = array_get_block( A, size_t x0, size_t x1, size_t fX=1,
                               size_t y0=0, size_t y1=0, size_t fY=1,
                               size_t z0=0, size_t z1=0, size_t fZ=1,
                               size_t t0=0, size_t t1=0, size_t fT=1 )
    """
    cdef fff_array *a, *b
    cdef fff_array asub
    a = fff_array_fromPyArray(A)
    asub = fff_array_get_block(a, x0, x1, fX, y0, y1, fY, z0, z1, fZ, t0, t1, fT)
    b = fff_array_new(asub.datatype, asub.dimX, asub.dimY, asub.dimZ, asub.dimT)
    fff_array_copy(b, &asub)
    B = fff_array_toPyArray(b)
    fff_array_delete(a)
    return B


def array_add(A, B): 
    """
    C = A + B 
    """
    cdef fff_array *a, *b, *c 
    
    a = fff_array_fromPyArray(A)
    b = fff_array_fromPyArray(B)
    c = fff_array_new(a.datatype, a.dimX, a.dimY, a.dimZ, a.dimT)
    fff_array_copy(c, a)
    fff_array_add(c, b) 
    C = fff_array_toPyArray(c)
    fff_array_delete(a)
    fff_array_delete(b)
    return C 


def array_mul(A, B): 
    """
    C = A * B 
    """
    cdef fff_array *a, *b, *c 
    
    a = fff_array_fromPyArray(A)
    b = fff_array_fromPyArray(B)
    c = fff_array_new(a.datatype, a.dimX, a.dimY, a.dimZ, a.dimT)
    fff_array_copy(c, a)
    fff_array_mul(c, b) 
    C = fff_array_toPyArray(c)
    fff_array_delete(a)
    fff_array_delete(b)
    return C 


def array_sub(A, B): 
    """
    C = A - B 
    """
    cdef fff_array *a, *b, *c 
    
    a = fff_array_fromPyArray(A)
    b = fff_array_fromPyArray(B)
    c = fff_array_new(a.datatype, a.dimX, a.dimY, a.dimZ, a.dimT)
    fff_array_copy(c, a)
    fff_array_sub(c, b) 
    C = fff_array_toPyArray(c)
    fff_array_delete(a)
    fff_array_delete(b)
    return C 


def array_div(A, B): 
    """
    C = A / B 
    """
    cdef fff_array *a, *b, *c 
    
    a = fff_array_fromPyArray(A)
    b = fff_array_fromPyArray(B)
    c = fff_array_new(a.datatype, a.dimX, a.dimY, a.dimZ, a.dimT)
    fff_array_copy(c, a)
    fff_array_div(c, b) 
    C = fff_array_toPyArray(c)
    fff_array_delete(a)
    fff_array_delete(b)
    return C 
