#include "polyaffine.h"

#include <math.h>
#include <stdlib.h>

#define TINY 1e-200


static double _gaussian(double* xyz, double* center, double* sigma)
{
  double aux, d2 = 0.0; 
  int i; 

  for (i=0; i<3; i++) {
    aux = xyz[i] - center[i]; 
    aux /= sigma[i]; 
    d2 += aux*aux;
  }

  return exp(-.5*d2);
}

/* Compute: y += w*x */
static void _add_weighted_affine(double* y, const double* x, double w)
{
  int i; 

  for (i=0; i<12; i++)
    y[i] += w*x[i]; 

  return; 
}

/* Compute: y = mat*x */ 
static void _apply_affine(double *y,  const double* mat, const double* x, double W)
{
  y[0] = mat[0]*x[0]+mat[1]*x[1]+mat[2]*x[2]+mat[3]; 
  y[1] = mat[4]*x[0]+mat[5]*x[1]+mat[6]*x[2]+mat[7]; 
  y[2] = mat[8]*x[0]+mat[9]*x[1]+mat[10]*x[2]+mat[11]; 

  if (W<TINY)
    W = TINY; 
  
  y[0] /= W; 
  y[1] /= W; 
  y[2] /= W; 

  return; 
} 

/*
  XYZ assumed contiguous double (N, 3)
  Centers assumed contiguous double (K, 3)
  Affines assumed contiguous double (K, 12)
 */


void apply_polyaffine(PyArrayObject* XYZ, 
		      const PyArrayObject* Centers, 
		      const PyArrayObject* Affines, 
		      const PyArrayObject* Sigma)

{

  PyArrayIterObject *iter_xyz, *iter_centers, *iter_affines;
  int axis = 1; 
  double *xyz, *center, *affine, *sigma; 
  double w, W; 
  double mat[12], t_xyz[3]; 
  size_t bytes_mat = 12*sizeof(double); 
  size_t bytes_xyz = 3*sizeof(double); 

  /* Initialize arrays and iterators */ 
  sigma = PyArray_DATA(Sigma);
  iter_xyz = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)XYZ, &axis);
  iter_centers = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)Centers, &axis);
  iter_affines = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)Affines, &axis);
    
  /* Loop over input points */ 
  while(iter_xyz->index < iter_xyz->size) {

    xyz = PyArray_ITER_DATA(iter_xyz);
    PyArray_ITER_RESET(iter_centers);
    PyArray_ITER_RESET(iter_affines);
    memset((void*)mat, 0, bytes_mat); 
    W = 0.0; 

    /* Loop over centers */
    while(iter_centers->index < iter_centers->size) {
      center = PyArray_ITER_DATA(iter_centers);
      affine = PyArray_ITER_DATA(iter_affines);
      w = _gaussian(xyz, center, sigma); 
      W += w; 
      _add_weighted_affine(mat, affine, w); 
      PyArray_ITER_NEXT(iter_centers); 
      PyArray_ITER_NEXT(iter_affines); 
    }

    /* Apply matrix */ 
    _apply_affine(t_xyz, mat, xyz, W); 
    memcpy((void*)xyz, (void*)t_xyz, bytes_xyz); 

    /* Update xyz iterator */ 
    PyArray_ITER_NEXT(iter_xyz); 
  }

  /* Free memory */ 
  Py_XDECREF(iter_xyz);
  Py_XDECREF(iter_centers);
  Py_XDECREF(iter_affines);

  return; 
}



