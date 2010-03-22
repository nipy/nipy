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



static inline void _soft_vote(double* res, int K, size_t pos, 
			      const double* ppm_data)
{
  size_t p = pos;
  int k;
  double* buf_res = res;
  
  for (k=0, buf_res=res; k<K; k++, buf_res++, p++)
    *buf_res += ppm_data[p];
  
  return;
}


static inline void _hard_vote(double* res, int K, size_t pos, 
			      const double* ppm_data)
{
  size_t p = pos;
  int k, kmax = -1;
  double max = 0, aux;
  
  for (k=0; k<K; k++, p++) {
    aux = ppm_data[p];
    if (aux>max)
      kmax = k;
    max = aux;
  }
  if (kmax >= 0)
    res[kmax] += 1;
  
  return;
}

/*
  
  ppm assumed contiguous double (X, Y, Z, K) 

  res assumed preallocated with size >= K 

*/

static void _ngb26_vote(double* res, 
                        const PyArrayObject* ppm,  
                        int x,
                        int y, 
                        int z,
                        void* vote_fn)
{
  int j = 0, xn, yn, zn, nn = 26, K = ppm->dimensions[3]; 
  int* buf_ngb; 
  const double* ppm_data = (double*)ppm->data; 
  size_t u3 = K; 
  size_t u2 = ppm->dimensions[2]*u3; 
  size_t u1 = ppm->dimensions[1]*u2;
  size_t pos; 
  void (*vote)(double*,int,size_t,const double*) = vote_fn;
  
  /*  Re-initialize output array */
  memset ((void*)res, 0, K*sizeof(double));
  
  /* Loop over neighbors */ 
  buf_ngb = ngb26; 
  while (j < nn) {
    xn = x + *buf_ngb; buf_ngb++; 
    yn = y + *buf_ngb; buf_ngb++;
    zn = z + *buf_ngb; buf_ngb++;
    pos = xn*u1 + yn*u2 + zn*u3;
    vote(res, K, pos, ppm_data);
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
		double beta,
                int copy,
                int hard)

{
  int npts, k, K, kk, x, y, z;
  double *p, *buf;
  double psum, tmp;  
  PyArrayIterObject* iter;
  int axis = 0; 
  double* ppm_data;
  size_t u3 = ppm->dimensions[3]; 
  size_t u2 = ppm->dimensions[2]*u3; 
  size_t u1 = ppm->dimensions[1]*u2;
  const double* lik_data = (double*)lik->data;
  size_t v1 = lik->dimensions[1];
  const int* XYZ_data = (int*)XYZ->data;
  size_t w1 = XYZ->dimensions[1], two_w1=2*w1;
  void (*vote)(double*,int,size_t,const double*);
  
  /* Dimensions */
  npts = PyArray_DIM((PyArrayObject*)XYZ, 1);
  K = PyArray_DIM((PyArrayObject*)ppm, 3);
    
  /* Copy or not copy */
  if (copy) {
    size_t S = PyArray_SIZE(ppm);
    ppm_data = (double*)calloc(S, sizeof(double));
    if (ppm_data==NULL) {
      fprintf(stderr, "Cannot allocate ppm copy\n"); 
      return; 
    }
    (double*)memcpy((void*)ppm_data, (void*)ppm->data, S*sizeof(double));
  }
  else
    ppm_data = (double*)ppm->data;
  
  /* Hard or soft vote */
  if (hard)
    vote = &_hard_vote;
  else
    vote = &_soft_vote;
  
  /* Allocate auxiliary vector */
  p = (double*)calloc(K, sizeof(double)); 
  
  /* Loop over points */ 
  iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)XYZ, &axis);

  /* Loop over voxels */ 
  while(iter->index < iter->size) {
    
    /* Compute the average ppm in the neighborhood */ 
    x = XYZ_data[iter->index];
    y = XYZ_data[w1+iter->index];
    z = XYZ_data[two_w1+iter->index]; 
    _ngb26_vote(p, ppm, x, y, z, (void*)vote); 
    
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
  
  /* Free memory */ 
  free(p);
  Py_XDECREF(iter);
  if (copy)
    free(ppm_data);
    
  return; 
}
