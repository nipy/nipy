#include <Python.h>
#include <numpy/arrayobject.h>
#include <fff_vector.h>
#include <fff_matrix.h>
#include <fff_array.h>


/*!
  \file fffpy.h
  \brief Python interface to \a fff 
  \author Alexis Roche, Benjamin Thyreau, Bertrand Thirion
  \date 2006-2009
*/

#ifndef NPY_VERSION
#define npy_intp intp
#define NPY_OWNDATA OWNDATA
#define NPY_CONTIGUOUS CONTIGUOUS
#define NPY_BEHAVED BEHAVED_FLAGS
#endif

#define fffpyZeroLONG() (PyArrayObject*)PyArray_SimpleNew(1,(npy_intp*)"\0\0\0\0", PyArray_LONG);



/*!
   \brief Import numpy C API

   Any Python module written in C, and using the fffpy interface, must
   call this function to work, because \c PyArray_API is defined
   static, in order not to share that symbol within the
   dso. (import_array() asks the pointer value to the python process)
*/
extern void fffpy_import_array(void);


/*!
  \brief Convert \c PyArrayObject to \c fff_vector 
  \param x input numpy array 
  
  This function may be seen as a \c fff_vector constructor compatible
  with \c fff_vector_delete. If the input has type \c PyArray_DOUBLE,
  whether or not it is contiguous, the new \c fff_vector is not
  self-owned and borrows a reference to the PyArrayObject's
  data. Otherwise, data are copied and the \c fff_vector is
  self-owned (hence contiguous) just like when created from
  scratch. Notice, the function returns \c NULL if the input array
  has more than one dimension.
*/ 
extern fff_vector* fff_vector_fromPyArray(const PyArrayObject* x); 

/*!
  \brief Convert \c fff_vector to \c PyArrayObject 
  \param y input vector

  Conversely to \c fff_vector_fromPyArray, this function acts as a \c
  fff_vector destructor compatible with \c fff_vector_new, returning
  a new PyArrayObject reference. If the input vector is contiguous and
  self-owned, array ownership is simply transferred to Python;
  otherwise, the data array is copied. 
*/ 
extern PyArrayObject* fff_vector_toPyArray(fff_vector* y); 

/*!
  \brief Convert \c fff_vector to \c PyArrayObject, without destruction
  \param y input const vector

  Unlike \c fff_vector_toPyArray, this function does not delete the
  input fff_vector. It always forces a copy of the data array. This
  function is useful when exporting to Python a fff_vector that
  belongs to a local structure having its own destruction method.
*/ 
extern PyArrayObject* fff_vector_const_toPyArray(const fff_vector* y); 

/*!
  \brief Convert \c PyArrayObject to \c fff_matrix 
  \param x input numpy array 
  
  This function may be seen as a \c fff_matrix constructor compatible
  with \c fff_matrix_free. If the input has type \c PyArray_DOUBLE and
  is contiguous, the new \c fff_matrix is not self-owned and borrows a
  reference to the PyArrayObject's data. Otherwise, data are copied
  and the \c fff_matrix is self-owned (hence contiguous) just like
  when created from scratch. \c NULL is returned if the input array
  does not have exactly two dimensions.

  Remarks: 1) non-contiguity provokes a copy because the \c fff_matrix
  structure does not support strides; 2) matrices in column-major
  order (Fortran convention) always get copied using this function.
*/  
extern fff_matrix* fff_matrix_fromPyArray(const PyArrayObject* x);

/*!
  \brief Convert \c fff_matrix to \c PyArrayObject 
  \param y input matrix
  
  Conversely to \c fff_matrix_fromPyArray, this function acts as a \c
  fff_matrix destructor compatible with \c fff_matrix_new, returning
  a new PyArrayObject reference. If the input matrix is contiguous and
  self-owned, array ownership is simply transferred to Python;
  otherwise, the data array is copied. 
*/  
extern PyArrayObject* fff_matrix_toPyArray(fff_matrix* y); 

/*!
  \brief Convert \c fff_matrix to \c PyArrayObject, without destruction
  \param y input const matrix

  Unlike \c fff_matrix_toPyArray, this function does not delete the
  input fff_matrix. It always forces a copy of the data array. This
  function is useful when exporting to Python a fff_matrix that
  belongs to a local structure having its own destruction method.
*/ 
extern PyArrayObject* fff_matrix_const_toPyArray(const fff_matrix* y); 



/*!
  \brief Maps a numpy array to an fff_array  
  \param x input array 

  This function instantiates an fff_array that borrows data from the
  numpy array. Delete using  \c fff_array_delete.

*/
extern fff_array* fff_array_fromPyArray(const PyArrayObject* x); 
extern PyArrayObject* fff_array_toPyArray(fff_array* y); 

extern fff_datatype fff_datatype_fromNumPy(int npy_type); 
extern int fff_datatype_toNumPy(fff_datatype fff_type); 

extern void fff_vector_fetch_using_NumPy(fff_vector* y, const char* data, npy_intp stride, int type, int itemsize);


/*
  Multi-iterator object. 
 */

typedef struct {
  
  int narr;
  int axis; 
  fff_vector** vector; 
  size_t index; 
  size_t size; 
  PyArrayMultiIterObject *multi;

} fffpy_multi_iterator;

extern fffpy_multi_iterator* fffpy_multi_iterator_new(int narr, int axis, ...); 
extern void fffpy_multi_iterator_delete(fffpy_multi_iterator* thisone); 
extern void fffpy_multi_iterator_update(fffpy_multi_iterator* thisone); 
extern void fffpy_multi_iterator_reset(fffpy_multi_iterator* thisone); 
