#include "mrf.h"

#include <math.h>
#include <stdlib.h>


/* Numpy import */
void mrf_import_array(void) { 
  import_array(); 
  return;
}

/*
  Compute the mean value of a vector image in the 26-neighborhood
  of a given element with indices (x,y,z). 
*/

int ngb26 [] = {1,0,0,
		-1,0,0,
		0,1,0,
		0,-1,0,
		1,1,0,
		-1,-1,0,
		1,-1,0,
		-1,1,0, 
		1,0,1,
		-1,0,1,
		0,1,1,
		0,-1,1, 
		1,1,1,
		-1,-1,1,
		1,-1,1,
		-1,1,1, 
		1,0,-1,
		-1,0,-1,
		0,1,-1,
		0,-1,-1, 
		1,1,-1,
		-1,-1,-1,
		1,-1,-1,
		-1,1,-1, 
		0,0,1,
		0,0,-1}; 


/*

  ppm assumed contiguous double [x,y,z,k] with K = 26 

*/

static void _ngb26_average(double* res, 
			   int dim_res,
			   const PyArrayObject* ppm,  
			   int x,
			   int y, 
			   int z)
{
  int j = 0, k, xn, yn, zn;
  unsigned int nn = 26;
  double *buf; 
  int* buf_ngb; 
  const double* ppm_data = (double*)ppm->data; 
  size_t u3 = ppm->dimensions[3]; 
  size_t u2 = ppm->dimensions[2]*u3; 
  size_t u1 = ppm->dimensions[1]*u2;

  /*  Re-initialize output array */
  for (k=0, buf=res; k<dim_res; k++, buf++)
    *buf = 0.0; 

  /* Loop over neighbors */ 
  buf_ngb = ngb26; 
  while (j < nn) {
    xn = x + *buf_ngb; buf_ngb++; 
    yn = y + *buf_ngb; buf_ngb++;
    zn = z + *buf_ngb; buf_ngb++;
    for (k=0, buf=res; k<dim_res; k++, buf++)
      *buf += ppm_data[xn*u1 + yn*u2 + zn*u3 + k];
    j ++; 
  }

  return; 
}

/*
  ppm assumed contiguous double (X, Y, Z, K) 

  lik assumed contiguous double (NPTS, K)

  XYZ assumed contiguous usigned int (3, NPTS)

 */

#define TINY 1e-20
void smooth_ppm(PyArrayObject* ppm, 
		const PyArrayObject* lik,
		const PyArrayObject* XYZ, 
		double beta)

{
  int npts, k, K, kk, x, y, z;
  double *p, *buf;
  double psum, tmp;  
  PyArrayIterObject* iter;
  int axis = 0; 
  double* ppm_data = (double*)ppm->data;
  size_t u3 = ppm->dimensions[3]; 
  size_t u2 = ppm->dimensions[2]*u3; 
  size_t u1 = ppm->dimensions[1]*u2;
  const double* lik_data = (double*)lik->data;
  size_t v1 = lik->dimensions[1];
  const int* XYZ_data = (int*)XYZ->data;
  size_t w1 = XYZ->dimensions[1], two_w1=2*w1;
   
  /* Dimensions */
  npts = PyArray_DIM((PyArrayObject*)XYZ, 1);
  K = PyArray_DIM((PyArrayObject*)ppm, 3);
  
  /* Allocate auxiliary vector */
  p = (double*)calloc(K, sizeof(double)); 

  /* Loop over points */ 
  iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)XYZ, &axis);
  while(iter->index < iter->size) {

    /* Compute the average ppm in the neighborhood */ 
    x = XYZ_data[iter->index];
    y = XYZ_data[w1+iter->index];
    z = XYZ_data[two_w1+iter->index]; 
    _ngb26_average(p, K, ppm, x, y, z); 

    /* Apply exponential transformation and multiply with likelihood
       term */
    psum = 0.0; 
    for (k=0, kk=(iter->index)*v1, buf=p; k<K; k++, kk++, buf++) {
      tmp = exp(beta*(*buf)) * lik_data[kk];
      psum += tmp; 
      *buf = tmp; 
    }

    /* Normalize to unitary sum */
    kk = x*u1 + y*u2 + z*u3; 
    if (psum > TINY) 
      for (k=0, buf=p; k<K; k++, kk++, buf++)
	ppm_data[kk] = *buf/psum; 
    else
      for (k=0, buf=p; k<K; k++, kk++, buf++)
	ppm_data[kk] = *buf; 

    /* Update iterator */ 
    PyArray_ITER_NEXT(iter); 

  }
  
  /* Free auxiliary vector */ 
  free(p);
  
  return; 
}
