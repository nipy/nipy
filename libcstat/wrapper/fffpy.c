#include "fffpy.h"
#include <stdarg.h>
#include <errno.h>

#define COPY_BUFFERS_USING_NUMPY 1


/* This function must be called before the module can work
   because PyArray_API is defined static, in order not to share that symbol
   within the dso. (import_array() asks the pointer value to the python process)
*/
void fffpy_import_array(void) { 
  import_array(); 
  return;
}


/* Static functions */
static npy_intp _PyArray_main_axis(const PyArrayObject* x, int* ok); 
static fff_vector* _fff_vector_new_from_buffer(const char* data, npy_intp dim, npy_intp stride, int type, int itemsize);
static fff_vector* _fff_vector_new_from_PyArrayIter(const PyArrayIterObject* it, npy_intp axis);
static void _fff_vector_sync_with_PyArrayIter(fff_vector* y, const PyArrayIterObject* it, npy_intp axis); 


/* Routines for copying 1d arrays into contiguous double arrays */ 
#if COPY_BUFFERS_USING_NUMPY 
# define COPY_BUFFER(y, data, stride, type, itemsize)	\
  fff_vector_fetch_using_NumPy(y, data, stride, type, itemsize); 
#else 
# define COPY_BUFFER(y, data, stride, type, itemsize)	\
  fff_vector_fetch(y, (void*)data, fff_datatype_fromNumPy(type), stride/itemsize) 
#endif 



/* 
   Copy a buffer using numpy. 

   Copy buffer x into y assuming that y is contiguous. 
*/ 
void fff_vector_fetch_using_NumPy(fff_vector* y, const char* x, npy_intp stride, int type, int itemsize) 
{
  npy_intp dim[1] = {(npy_intp)y->size}; 
  npy_intp strides[1] = {stride};   
  PyArrayObject* X = (PyArrayObject*) PyArray_New(&PyArray_Type, 1, dim, type, strides, 
						  (void*)x, itemsize, NPY_BEHAVED, NULL); 
  PyArrayObject* Y = (PyArrayObject*) PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, (void*)y->data);
  PyArray_CastTo(Y, X); 
  Py_XDECREF(Y);
  Py_XDECREF(X);
  return; 
}

/* 
   Create a fff_vector from an already allocated buffer. This function
   acts as a fff_vector constructor that is compatible with
   fff_vector_delete.
*/

static fff_vector* _fff_vector_new_from_buffer(const char* data, npy_intp dim, npy_intp stride, int type, int itemsize)
{
  fff_vector* y; 
  size_t sizeof_double = sizeof(double); 

  /* If the input array is double and is aligned, just wrap without copying */
  if ((type == NPY_DOUBLE) && (itemsize==sizeof_double)) {
    y = (fff_vector*)malloc(sizeof(fff_vector)); 
    y->size = (size_t)dim;
    y->stride = (size_t)stride/sizeof_double;
    y->data = (double*)data;
    y->owner = 0; 
  }
  /* Otherwise, output a owner contiguous vector with copied data */
  else {
    y = fff_vector_new((size_t)dim); 
    COPY_BUFFER(y, data, stride, type, itemsize); 
  }

  return y; 
}


/* Find the axis with largest dimension */ 
npy_intp _PyArray_main_axis(const PyArrayObject* x, int* ok)
{
  npy_intp axis, count, i, dim, ndim = PyArray_NDIM(x); 
  *ok = 1; 

  axis = 0; 
  count = 0; 
  for(i=0; i<ndim; i++) {
    dim = PyArray_DIM(x,i); 
    if (dim > 1) {
      count ++; 
      axis = i;
    }
  }

  if (count > 1) 
    *ok = 0; 

  return axis;
}

fff_vector* fff_vector_fromPyArray(const PyArrayObject* x) 
{
  fff_vector* y;
  int ok;  
  npy_intp axis = _PyArray_main_axis(x, &ok);

  if (!ok) {
    FFF_ERROR("Input array is not a vector", EINVAL);
    return NULL;
  }

  y = _fff_vector_new_from_buffer(PyArray_DATA(x),
				  PyArray_DIM(x, axis),
				  PyArray_STRIDE(x, axis),
				  PyArray_TYPE(x), 
				  PyArray_ITEMSIZE(x));
  return y; 
}


/*
  Export a fff_vector to a PyArray, and delete it. This function is a
  fff_vector destructor compatible with any either fff_vector_new or
  _fff_vector_new_from_buffer.
*/ 
PyArrayObject* fff_vector_toPyArray(fff_vector* y) 
{
  PyArrayObject* x; 
  size_t size;
  npy_intp dims[1]; 
   if (y == NULL) 
    return NULL;
   size = y->size;

  dims[0] = (npy_intp) size;  
 
  /* If the fff_vector is owner (hence contiguous), just pass the
     buffer to Python and transfer ownership */ 
  if (y->owner) {
    x = (PyArrayObject*) PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void*)y->data);
    x->flags = (x->flags) | NPY_OWNDATA; 
  }
  /* Otherwise, create Python array from scratch */ 
  else 
    x = fff_vector_const_toPyArray(y); 
 
  /* Ciao bella */ 
  free(y);

  return x;
}

/* Export without deleting */ 
PyArrayObject* fff_vector_const_toPyArray(const fff_vector* y)
{
  PyArrayObject* x;
  size_t i, size = y->size, stride = y->stride; 
  double* data = (double*) malloc(size*sizeof(double)); 
  double* bufX = data; 
  double* bufY = y->data; 
  npy_intp dims[1]; 
  
  dims[0] = (npy_intp) size;  
  for (i=0; i<size; i++, bufX++, bufY+=stride) 
    *bufX = *bufY; 
  x = (PyArrayObject*) PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void*)data);
  x->flags = (x->flags) | NPY_OWNDATA; 

  return x; 
}



/* 
   Get a fff_matrix from an input PyArray. This function acts as a
   fff_vector constructor that is compatible with fff_vector_delete.
*/ 
fff_matrix* fff_matrix_fromPyArray(const PyArrayObject* x) 
{
  fff_matrix* y;
  npy_intp dim[2];
  PyArrayObject* xd;

  /* Check that the input object is a two-dimensional array */ 
  if (PyArray_NDIM(x) != 2) {
    FFF_ERROR("Input array is not a matrix", EINVAL);
    return NULL;
  }


  /* If the PyArray is double, contiguous and aligned just wrap without
     copying */
  if ((PyArray_TYPE(x) == NPY_DOUBLE) && 
       (PyArray_ISCONTIGUOUS(x)) && 
       (PyArray_ISALIGNED(x))) {
    y = (fff_matrix*) malloc(sizeof(fff_matrix)); 
    y->size1 = (size_t) PyArray_DIM(x,0);
    y->size2 = (size_t) PyArray_DIM(x,1);
    y->tda = y->size2; 
    y->data = (double*) PyArray_DATA(x);
    y->owner = 0;
  }
  /* Otherwise, output a owner (contiguous) matrix with copied
     data */
  else {
    size_t dim0 = PyArray_DIM(x,0), dim1 = PyArray_DIM(x,1);
    y = fff_matrix_new((size_t)dim0, (size_t)dim1); 
    dim[0] = dim0;  
    dim[1] = dim1;

    xd = (PyArrayObject*) PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, (void*)y->data);
    PyArray_CastTo(xd, (PyArrayObject*)x); 
    Py_XDECREF(xd);
  }
  
  return y;
}


/*
  Export a fff_matrix to a PyArray, and delete it. This function is a
  fff_matrix destructor compatible with any of the following
  constructors: fff_matrix_new and fff_matrix_fromPyArray.
*/ 
PyArrayObject* fff_matrix_toPyArray(fff_matrix* y) 
{
  PyArrayObject* x; 
  size_t size1;
  size_t size2;
  size_t tda;
  npy_intp dims[2]; 
  if (y == NULL) 
    return NULL;
  size1 = y->size1;
  size2 = y->size2;
  tda = y->tda; 

  dims[0] = (npy_intp) size1;  
  dims[1] = (npy_intp) size2;  
  
  /* If the fff_matrix is contiguous and owner, just pass the
     buffer to Python and transfer ownership */ 
  if ((tda == size2) && (y->owner)) {
    x = (PyArrayObject*) PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, (void*)y->data);
    x->flags = (x->flags) | NPY_OWNDATA; 
  }
  /* Otherwise, create PyArray from scratch. Note, the input
     fff_matrix is necessarily in row-major order. */ 
  else 
    x = fff_matrix_const_toPyArray(y); 
  
  /* Ciao bella */ 
  free(y);
   
  return x;
}


/* Export without deleting */
PyArrayObject* fff_matrix_const_toPyArray(const fff_matrix* y)
{
  PyArrayObject* x;
  size_t size1 = y->size1, size2 = y->size2, tda = y->tda; 
  size_t i, j, pos;
  double* data = (double*) malloc(size1*size2*sizeof(double)); 
  double* bufX = data;
  double* bufY = y->data; 
  npy_intp dims[2]; 

  dims[0] = (npy_intp) size1;  
  dims[1] = (npy_intp) size2;  
  for (i=0; i<size1; i++) {
    pos = tda*i;
    for (j=0; j<size2; j++, bufX++, pos++)
      *bufX = bufY[pos];
  }
  
  x = (PyArrayObject*) PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, (void*)data);
  x->flags = (x->flags) | NPY_OWNDATA; 

  return x;  
}

/** Static routines **/ 



/**** Data type conversions *****/ 
fff_datatype fff_datatype_fromNumPy(int npy_type)
{

  fff_datatype fff_type; 

  switch (npy_type) {
  case NPY_UBYTE:
    fff_type = FFF_UCHAR; 
    break;
  case NPY_BYTE:
    fff_type = FFF_SCHAR;
    break;
  case NPY_USHORT: 
    fff_type = FFF_USHORT;
    break;
  case NPY_SHORT: 
    fff_type = FFF_SSHORT;
    break;
  case NPY_UINT: 
    fff_type = FFF_UINT;
    break;
  case NPY_INT: 
    fff_type = FFF_INT;
    break;
  case NPY_ULONG: 
    fff_type = FFF_ULONG;
    break;
  case NPY_LONG:
    fff_type = FFF_LONG;
    break;
  case NPY_FLOAT: 
    fff_type = FFF_FLOAT;
    break;
  case NPY_DOUBLE: 
    fff_type = FFF_DOUBLE;
    break;
  default: 
    fff_type = FFF_UNKNOWN_TYPE;
    break; 
  }

  /* Return the datatype */ 
  return fff_type; 
}

int fff_datatype_toNumPy(fff_datatype fff_type)
{
  int npy_type; 

  switch(fff_type) {
  case FFF_UCHAR:
    npy_type = NPY_UBYTE; 
    break;
  case FFF_SCHAR:
    npy_type = NPY_BYTE;
    break;
  case FFF_USHORT: 
    npy_type = NPY_USHORT;
    break;
  case FFF_SSHORT: 
    npy_type = NPY_SHORT;
    break;
  case FFF_UINT: 
    npy_type = NPY_UINT;
    break;
  case FFF_INT: 
    npy_type = NPY_INT;
    break;
  case FFF_ULONG: 
    npy_type = NPY_ULONG;
    break;
  case FFF_LONG:
    npy_type = NPY_LONG;
    break;
  case FFF_FLOAT: 
    npy_type = NPY_FLOAT;
    break;
  case FFF_DOUBLE: 
    npy_type = NPY_DOUBLE;
    break;
  default: 
    npy_type = NPY_NOTYPE;
    break; 
  }
  return npy_type; 
}

/**** fff_array interface ****/

fff_array* fff_array_fromPyArray(const PyArrayObject* x) 
{
  fff_array* y;
  fff_datatype datatype; 
  unsigned int nbytes; 
  size_t dimX = 1, dimY = 1, dimZ = 1, dimT = 1; 
  size_t offX = 0, offY = 0, offZ = 0, offT = 0; 
  size_t ndims = (size_t)PyArray_NDIM(x);

  /* Check that the input array has less than four dimensions */ 
  if (ndims > 4) {
    FFF_ERROR("Input array has more than four dimensions", EINVAL);
    return NULL;
  }
  /* Check that the input array is aligned */ 
  if (! PyArray_ISALIGNED(x)) {
    FFF_ERROR("Input array is not aligned", EINVAL);
    return NULL;
  }
  /* Match the data type */
  datatype = fff_datatype_fromNumPy(PyArray_TYPE(x)); 
  if (datatype == FFF_UNKNOWN_TYPE) { 
    FFF_ERROR("Unrecognized data type", EINVAL);
    return NULL;    
  }
  
  /* Dimensions and offsets */ 
  nbytes = fff_nbytes(datatype); 
  dimX = PyArray_DIM(x, 0);
  offX = PyArray_STRIDE(x, 0)/nbytes;
  if (ndims > 1) {
    dimY = PyArray_DIM(x, 1);
    offY = PyArray_STRIDE(x, 1)/nbytes;
    if (ndims > 2) {
      dimZ = PyArray_DIM(x, 2);
      offZ = PyArray_STRIDE(x, 2)/nbytes;
      if (ndims > 3) {
	dimT = PyArray_DIM(x, 3);
	offT = PyArray_STRIDE(x, 3)/nbytes;
      }
    }
  }

  /* Create array (not owner) */ 
  y = (fff_array*)malloc(sizeof(fff_array)); 
  *y = fff_array_view(datatype, 
		      (void*) PyArray_DATA(x), 
		      dimX, dimY, dimZ, dimT, 
		      offX, offY, offZ, offT);
  
  return y;
}



PyArrayObject* fff_array_toPyArray(fff_array* y) 
{
  PyArrayObject* x; 
  npy_intp dims[4];
  int datatype; 
  fff_array* yy;
  if (y == NULL) 
    return NULL;
  dims[0] = y->dimX;
  dims[1] = y->dimY;
  dims[2] = y->dimZ;
  dims[3] = y->dimT;

  /* Match data type */
  datatype = fff_datatype_toNumPy(y->datatype); 
  if (datatype == NPY_NOTYPE) { 
    FFF_ERROR("Unrecognized data type", EINVAL);
    return NULL;    
  }
    
  /* Make sure the fff array owns its data, which may require a copy */ 
  if (y->owner) 
    yy = y; 
  else {
    yy = fff_array_new(y->datatype, y->dimX, y->dimY, y->dimZ, y->dimT); 
    fff_array_copy(yy, y); 
  }
  /* 
     Create a Python array from the array data (which is contiguous
     since it is owner).  We can use PyArray_SimpleNewFromData given
     that yy is C-contiguous by fff_array_new.  
  */ 
   x = (PyArrayObject*) PyArray_SimpleNewFromData(yy->ndims, dims, datatype, (void*)yy->data);

  /* Transfer ownership to Python */ 
  x->flags = (x->flags) | NPY_OWNDATA;
  
  /* Dealloc memory if needed */ 
  if (! y->owner)
    free(yy); 

  /* Delete array */ 
  free(y); 
  return x;
}






/********************************************************************

  Multi-iterator object. 

 ********************************************************************/

static int _PyArray_BroadcastAllButAxis (PyArrayMultiIterObject* mit, int axis); 


/* 
   Create a fff multi iterator object. 

   Involves creating a PyArrayMultiArrayIter instance that lets us
   iterate simultaneously on an arbitrary number of numpy arrays
   EXCEPT in one common axis.

   There does not seem to exist a built-in PyArrayMultiArrayIter
   constructor for this usage. If it pops up one day, part of the
   following code should be replaced. 

   Similarly to the default PyArrayMultiArrayIter constructor, we need
   to set up broadcasting rules. For now, we simply impose that all
   arrays have exactly the same number of dimensions and that all
   dimensions be equal except along the "non-iterated" axis.

   FIXME: The following code does not perform any checking, and will
   surely crash if the arrays do not fulfill the conditions.
*/ 

fffpy_multi_iterator* fffpy_multi_iterator_new(int narr, int axis, ...)
{
  fffpy_multi_iterator* thisone; 
  va_list va;
  fff_vector** vector; 
  PyArrayMultiIterObject *multi;
  PyObject *current, *arr;
  int i, err=0;

  /* Create new instance */ 
  thisone = (fffpy_multi_iterator*)malloc(sizeof(fffpy_multi_iterator));
  multi = PyArray_malloc(sizeof(PyArrayMultiIterObject));
  vector = (fff_vector**)malloc(narr*sizeof(fff_vector*)); 

  /* Initialize the PyArrayMultiIterObject instance from the variadic arguments */ 
  PyObject_Init((PyObject *)multi, &PyArrayMultiIter_Type);
  
  for (i=0; i<narr; i++) 
    multi->iters[i] = NULL;
  multi->numiter = narr;
  multi->index = 0;
  
  va_start(va, axis);
  for (i=0; i<narr; i++) {
    current = va_arg(va, PyObject *);
    arr = PyArray_FROM_O(current);
    if (arr==NULL) {
      err=1; break;
    }
    else {
      multi->iters[i] = (PyArrayIterObject *)PyArray_IterAllButAxis(arr, &axis);
      Py_DECREF(arr);
    }
  }
  
  va_end(va);

  /* Test */
  if (!err && _PyArray_BroadcastAllButAxis(multi, axis) < 0) 
    err=1; 
  if (err) {
    FFF_ERROR("Cannot create broadcast object", ENOMEM); 
    free(thisone); 
    free(vector); 
    Py_DECREF(multi);
    return NULL;
  }
  
  /* Initialize the multi iterator */  
  PyArray_MultiIter_RESET(multi);

  /* Create the fff vectors (views or copies) */ 
  for(i=0; i<narr; i++) 
    vector[i] = _fff_vector_new_from_PyArrayIter((const PyArrayIterObject*)multi->iters[i], axis); 
  
  /* Instantiate fiels */ 
  thisone->narr = narr; 
  thisone->axis = axis; 
  thisone->vector = vector;
  thisone->multi = multi; 
  thisone->index = thisone->multi->index; 
  thisone->size = thisone->multi->size;   

  return thisone; 
}


void fffpy_multi_iterator_delete(fffpy_multi_iterator* thisone)
{
  unsigned int i; 

  Py_DECREF(thisone->multi);
  for(i=0; i<thisone->narr; i++) 
    fff_vector_delete(thisone->vector[i]);
  free(thisone->vector); 
  free(thisone); 
  return; 
}

void fffpy_multi_iterator_update(fffpy_multi_iterator* thisone)
{
  unsigned int i; 

  PyArray_MultiIter_NEXT(thisone->multi); 
  for(i=0; i<thisone->narr; i++) 
    _fff_vector_sync_with_PyArrayIter(thisone->vector[i], (const PyArrayIterObject*)thisone->multi->iters[i], thisone->axis); 
  thisone->index = thisone->multi->index; 
  return; 
}

void fffpy_multi_iterator_reset(fffpy_multi_iterator* thisone)
{
  unsigned int i; 

  PyArray_MultiIter_RESET(thisone->multi); 
  for(i=0; i<thisone->narr; i++) 
    _fff_vector_sync_with_PyArrayIter(thisone->vector[i], (const PyArrayIterObject*)thisone->multi->iters[i], thisone->axis); 
  thisone->index = thisone->multi->index; 
  return; 
}

static int _PyArray_BroadcastAllButAxis (PyArrayMultiIterObject* mit, int axis)
{
  int i, nd;
  npy_intp size, tmp;
  PyArrayIterObject *it;

  /* Not very robust */
  it = mit->iters[0];  

  /* Set the dimensions */
  nd = it->ao->nd; 
  mit->nd = nd;
  for(i=0, size=1; i<nd; i++) {
    tmp = it->ao->dimensions[i];
    mit->dimensions[i] = tmp; 
    if (i!=axis) 
      size *= tmp; 
  } 
  mit->size = size;   
  
  /* Not very robust either */ 
  return 0; 
}


/* Create an fff_vector from a PyArrayIter object */ 
fff_vector* _fff_vector_new_from_PyArrayIter(const PyArrayIterObject* it, npy_intp axis)
{
  fff_vector* y; 
  char* data = PyArray_ITER_DATA(it);
  PyArrayObject* ao = (PyArrayObject*) it->ao; 
  npy_intp dim = PyArray_DIM(ao, axis); 
  npy_intp stride = PyArray_STRIDE(ao, axis); 
  int type = PyArray_TYPE(ao); 
  int itemsize = PyArray_ITEMSIZE(ao); 
   
  y = _fff_vector_new_from_buffer(data, dim, stride, type, itemsize); 
  return y;
}


/* Fetch vector data from an iterator (view or copy) */ 
void _fff_vector_sync_with_PyArrayIter(fff_vector* y, const PyArrayIterObject* it, npy_intp axis) 
{
  if (y->owner) {
    PyArrayObject* ao = (PyArrayObject*) it->ao; 
    COPY_BUFFER(y, PyArray_ITER_DATA(it), PyArray_STRIDE(ao, axis),
		PyArray_TYPE(ao), PyArray_ITEMSIZE(ao));
  }
  else 
    y->data = (double*) PyArray_ITER_DATA(it); 
  
  return; 
}


