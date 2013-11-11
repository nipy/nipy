# -*- Mode: Python -*-  Not really, but the syntax is close enough

# :Author: 	Alexis Roche

# Include numpy defines via Cython
from numpy cimport ndarray, import_array, npy_intp

# Redefine size_t
ctypedef unsigned long int size_t


# Exports from fff_base.h
cdef extern from "fff_base.h":

    ctypedef enum fff_datatype:
        FFF_UNKNOWN_TYPE = -1, 
        FFF_UCHAR = 0,         
        FFF_SCHAR = 1,         
        FFF_USHORT = 2,        
        FFF_SSHORT = 3,        
        FFF_UINT = 4,          
        FFF_INT = 5,           
        FFF_ULONG = 6,         
        FFF_LONG = 7,          
        FFF_FLOAT = 8,         
        FFF_DOUBLE = 9
        
    unsigned int fff_nbytes(fff_datatype type) 

# Exports from fff_vector.h 
cdef extern from "fff_vector.h":
    
    ctypedef struct fff_vector:
        size_t size
        size_t stride
        int owner
        double* data

    fff_vector* fff_vector_new(size_t n)
    void fff_vector_delete(fff_vector* x)
    fff_vector fff_vector_view(double* data, size_t size, size_t stride)
    double fff_vector_get(fff_vector * x, size_t i)
    void fff_vector_set(fff_vector * x, size_t i, double a)
    void fff_vector_set_all(fff_vector * x, double a) 
    void fff_vector_scale(fff_vector * x, double a) 
    void fff_vector_add_constant(fff_vector * x, double a)
    void fff_vector_memcpy(fff_vector* x, fff_vector* y)
    void fff_vector_fetch(fff_vector* x, void* data, fff_datatype datatype, size_t stride) 
    void fff_vector_add(fff_vector * x, fff_vector * y)
    void fff_vector_sub(fff_vector * x, fff_vector * y)
    void fff_vector_mul(fff_vector * x, fff_vector * y)
    void fff_vector_div(fff_vector * x, fff_vector * y)
    long double fff_vector_sum(fff_vector* x)
    long double fff_vector_ssd(fff_vector* x, double* m, int fixed)
    long double fff_vector_sad(fff_vector* x, double m)
    double fff_vector_median(fff_vector* x)
    double fff_vector_quantile(fff_vector* x, double r, int interp)
    double fff_vector_wmedian_from_sorted_data(fff_vector* x_sorted, fff_vector* w)

# Exports from fff_matrix.h
cdef extern from "fff_matrix.h":

    ctypedef struct fff_matrix:
        size_t size1
        size_t size2
        size_t tda
        int owner
        double* data

    fff_matrix* fff_matrix_new(size_t nr, size_t nc)
    void fff_matrix_delete(fff_matrix* A)
    fff_matrix fff_matrix_view(double* data, size_t size1, size_t size2, size_t tda)
    double fff_matrix_get(fff_matrix* A, size_t i, size_t j)
    void fff_matrix_set_all(fff_matrix * A, double a)
    void fff_matrix_scale(fff_matrix * A, double a)
    void fff_matrix_add_constant(fff_matrix * A, double a)
    void fff_matrix_get_row(fff_vector * x, fff_matrix * A, size_t i)
    fff_matrix_get_col(fff_vector * x, fff_matrix * A, size_t j)
    fff_matrix_get_diag(fff_vector * x, fff_matrix * A)
    fff_matrix_set_row(fff_matrix * A, size_t i, fff_vector * x)
    fff_matrix_set_col(fff_matrix * A, size_t j, fff_vector * x)
    fff_matrix_set_diag(fff_matrix * A, fff_vector * x)
    void fff_matrix_transpose(fff_matrix* A, fff_matrix* B)
    void fff_matrix_memcpy(fff_matrix* A, fff_matrix* B)
    fff_matrix fff_matrix_view(double* data, size_t size1, size_t size2, size_t tda) 
    void fff_matrix_add (fff_matrix * A, fff_matrix * B)
    void fff_matrix_sub (fff_matrix * A, fff_matrix * B)
    void fff_matrix_mul_elements (fff_matrix * A, fff_matrix * B)
    void fff_matrix_div_elements (fff_matrix * A, fff_matrix * B)
                

# Exports from fff_array.h
cdef extern from "fff_array.h":

    ctypedef enum fff_array_ndims: 
        FFF_ARRAY_1D = 1,   
        FFF_ARRAY_2D = 2,   
        FFF_ARRAY_3D = 3,   
        FFF_ARRAY_4D = 4    
        
    ctypedef struct fff_array:
        fff_array_ndims ndims
        fff_datatype datatype
        size_t dimX
        size_t dimY
        size_t dimZ
        size_t dimT
        unsigned int offsetX
        unsigned int offsetY
        unsigned int offsetZ
        unsigned int offsetT
        void* data
        int owner

    fff_array* fff_array_new(fff_datatype datatype, size_t dimX, size_t dimY, size_t dimZ, size_t dimT)
    fff_array* fff_array_new1d(fff_datatype datatype, size_t dimX)
    fff_array* fff_array_new2d(fff_datatype datatype, size_t dimX, size_t dimY)
    fff_array* fff_array_new3d(fff_datatype datatype, size_t dimX, size_t dimY, size_t dimZ)
    void fff_array_delete(fff_array* thisone)
    double fff_array_get(fff_array* thisone, size_t x, size_t y, size_t z, size_t t)
    fff_array fff_array_get_block(fff_array* thisone,
                                  size_t x0, size_t x1, size_t fX,
                                  size_t y0, size_t y1, size_t fY,
                                  size_t z0, size_t z1, size_t fZ,
                                  size_t t0, size_t t1, size_t fT)
    fff_array fff_array_get_block1d(fff_array* thisone, size_t x0, size_t x1, size_t fX)
    fff_array fff_array_get_block2d(fff_array* thisone,
                                    size_t x0, size_t x1, size_t fX,
                                    size_t y0, size_t y1, size_t fY)
    fff_array fff_array_get_block3d(fff_array* thisone,
                                    size_t x0, size_t x1, size_t fX,
                                    size_t y0, size_t y1, size_t fY,
                                    size_t z0, size_t z1, size_t fZ)
    void fff_array_set(fff_array* thisone, size_t x, size_t y, size_t z, size_t t, double value)
    void fff_array_set1d(fff_array* thisone, size_t x, double value)
    void fff_array_set2d(fff_array* thisone, size_t x, size_t y, double value)
    void fff_array_set3d(fff_array* thisone, size_t x, size_t y, size_t z, double value)
    void fff_array_set_all(fff_array* thisone, double c)
    void fff_array_extrema(double* min, double* max, fff_array* thisone)
    void fff_array_copy(fff_array* ares,  fff_array* asrc)
    void fff_array_add(fff_array * x, fff_array * y)
    void fff_array_sub(fff_array * x, fff_array * y)
    void fff_array_div(fff_array * x, fff_array * y)
    void fff_array_mul(fff_array * x, fff_array * y)
    void fff_array_clamp(fff_array* ares, fff_array* asrc, double th, int* clamp)

# Exports from the Python fff wrapper
cdef extern from "fffpy.h":

    ctypedef struct fffpy_multi_iterator:
        int narr
        int axis
        fff_vector** vector 
        size_t index 
        size_t size 

    void fffpy_import_array()
    fff_vector* fff_vector_fromPyArray(ndarray x)
    ndarray fff_vector_toPyArray(fff_vector* y)
    ndarray fff_vector_const_toPyArray(fff_vector* y)
    fff_matrix* fff_matrix_fromPyArray(ndarray x)
    ndarray fff_matrix_toPyArray(fff_matrix* y)
    ndarray fff_matrix_const_toPyArray(fff_matrix* y)
    fff_array* fff_array_fromPyArray(ndarray x) 
    ndarray fff_array_toPyArray(fff_array* y) 
    fff_datatype fff_datatype_fromNumPy(int npy_type)
    int fff_datatype_toNumPy(fff_datatype fff_type)
    void fff_vector_fetch_using_NumPy(fff_vector* y, char* data, npy_intp stride, int type, int itemsize)
    fffpy_multi_iterator* fffpy_multi_iterator_new(int narr, int axis, ...) 
    void fffpy_multi_iterator_delete(fffpy_multi_iterator* thisone)
    void fffpy_multi_iterator_update(fffpy_multi_iterator* thisone)
    void fffpy_multi_iterator_reset(fffpy_multi_iterator* thisone)

    
