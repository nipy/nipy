# -*- Mode: Python -*-  Not really, but the syntax is close enough


"""
Iterators for testing. 
Author: Alexis Roche, 2009.
"""

__version__ = '0.1'


# Includes
from fff cimport *
from numpy cimport dtype
cimport numpy as cnp

# Initialize numpy
fffpy_import_array()
import_array()
import numpy as np


c_types = ['unknown type', 'unsigned char', 'signed char', 'unsigned short', 'signed short', 
           'int', 'unsigned int', 'unsigned long', 'long', 'float', 'double']

fff_types = [FFF_UNKNOWN_TYPE, FFF_UCHAR, FFF_SCHAR, FFF_USHORT, FFF_SSHORT, 
             FFF_UINT, FFF_INT, FFF_ULONG, FFF_LONG, FFF_FLOAT, FFF_DOUBLE]  

npy_types = [cnp.NPY_NOTYPE, cnp.NPY_UBYTE, cnp.NPY_BYTE, cnp.NPY_USHORT,
             cnp.NPY_SHORT, cnp.NPY_UINT, cnp.NPY_INT, cnp.NPY_ULONG,
             cnp.NPY_LONG, cnp.NPY_FLOAT, cnp.NPY_DOUBLE]


def fff_type(dtype T):
    """
    fff_t, nbytes = fff_type(T)

    T is a np.dtype instance. Return a tuple (str, int). 
    """
    cdef fff_datatype fff_t
    cdef unsigned int nbytes
    fff_t =  fff_datatype_fromNumPy(T.type_num)
    nbytes =  fff_nbytes(fff_t)
    return c_types[fff_types.index(<int>fff_t)], nbytes


def npy_type(T): 
    """
    npy_t, nbytes = npy_type(T)

    T is a string. Return a tuple (str, int). 
    """
    cdef int npy_t
    cdef fff_datatype fff_t
    cdef unsigned int nbytes
    fff_t = <fff_datatype>fff_types[c_types.index(T)]
    npy_t = fff_datatype_toNumPy(fff_t)
    nbytes =  fff_nbytes(fff_t)
    return c_types[npy_types.index(npy_t)], nbytes

def pass_vector(ndarray X):
    """
    Y = pass_vector(X)
    """
    cdef fff_vector *x, *y
    x = fff_vector_fromPyArray(X)
    y = fff_vector_new(x.size)
    fff_vector_memcpy(y, x)
    fff_vector_delete(x)
    return fff_vector_toPyArray(y)


def copy_vector(ndarray X, int flag): 
    """
    Y = copy_vector(X, flag)

    flag == 0 ==> use fff
    flag == 1 ==> use numpy
    """
    cdef fff_vector *y
    cdef void* data
    cdef int size, stride, relstride, type, itemsize
    cdef fff_datatype fff_type

    data = <void*>X.data 
    size = X.shape[0]
    stride = X.strides[0]
    itemsize = X.descr.elsize
    type = X.descr.type_num 
    
    relstride = stride/itemsize
    fff_type =  fff_datatype_fromNumPy(type)

    y = fff_vector_new(size)

    if flag == 0: 
        fff_vector_fetch(y, data, fff_type, relstride) 
    else: 
        fff_vector_fetch_using_NumPy(y, <char*>data, stride, type, itemsize)

    return fff_vector_toPyArray(y)


def pass_matrix(ndarray X):
    """
    Y = pass_matrix(X)
    """
    cdef fff_matrix *x, *y 
    x = fff_matrix_fromPyArray(X)
    y = fff_matrix_new(x.size1, x.size2)
    fff_matrix_memcpy(y, x)
    fff_matrix_delete(x)
    return fff_matrix_toPyArray(y)


def pass_array(ndarray X):
    """
    Y = pass_array(X)
    """
    cdef fff_array *x, *y
    x = fff_array_fromPyArray(X)
    y = fff_array_new(x.datatype, x.dimX, x.dimY, x.dimZ, x.dimT)
    fff_array_copy(y, x)
    fff_array_delete(x)
    return fff_array_toPyArray(y)


def pass_vector_via_iterator(ndarray X, int axis=0, int niters=0):
    """
    Y = pass_vector_via_iterator(X, axis=0, niters=0)
    """
    cdef fff_vector *x, *y, *z
    cdef fffpy_multi_iterator* multi

    Xdum = X.copy() ## at least two arrays needed for multi iterator
    multi = fffpy_multi_iterator_new(2, axis, <void*>X, <void*>Xdum)
    x = multi.vector[0]

    while(multi.index < niters): 
        fffpy_multi_iterator_update(multi)

    y = fff_vector_new(x.size)
    fff_vector_memcpy(y, x)
    fffpy_multi_iterator_delete(multi)
    return fff_vector_toPyArray(y)


def copy_via_iterators(ndarray Y, int axis=0): 
    """
    Z = copy_via_iterators(Y, int axis=0) 

    Copy array Y into Z via fff's PyArray_MultiIterAllButAxis C function.
    Behavior should be equivalent to Z = Y.copy(). 
    """
    cdef fff_vector *y, *z
    cdef fffpy_multi_iterator* multi
   
    # Allocate output array
    Z = np.zeros_like(Y, dtype=np.float)

    # Create a new array iterator 
    multi = fffpy_multi_iterator_new(2, axis, <void*>Y, <void*>Z)
    
    # Create views
    y = multi.vector[0]
    z = multi.vector[1]

    # Loop 
    while(multi.index < multi.size):
        fff_vector_memcpy(z, y)
        fffpy_multi_iterator_update(multi)
    
    # Free memory
    fffpy_multi_iterator_delete(multi)

    # Return
    return Z


def sum_via_iterators(ndarray Y, int axis=0): 
    """
    Z = dummy_iterator(Y, int axis=0) 

    Return the sum of input elements along the dimension specified by axis.
    Behavior should be equivalent to Z = Y.sum(axis). 
    """
    cdef fff_vector *y, *z
    cdef fffpy_multi_iterator* multi
   
    # Allocate output array
    dims = [Y.shape[i] for i in range(Y.ndim)]
    dims[axis] = 1
    Z = np.zeros(dims)

    # Create a new array iterator 
    multi = fffpy_multi_iterator_new(2, axis, <void*>Y, <void*>Z)

    # Create views
    y = multi.vector[0]
    z = multi.vector[1]

    # Loop 
    while(multi.index < multi.size):
        z.data[0] = <double>fff_vector_sum(y)
        fffpy_multi_iterator_update(multi)
    
    # Free memory
    fffpy_multi_iterator_delete(multi)

    # Return
    return Z.squeeze()


