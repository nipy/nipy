#include "mrf.h"

#include <math.h>
#include <stdlib.h>

#ifdef _MSC_VER
#define inline __inline
#endif


/* Numpy import */
void mrf_import_array(void) { 
  import_array(); 
  return;
}

/* Encode neighborhood systems using static arrays */
int ngb6 [] = {1,0,0,
	       -1,0,0,
	       0,1,0,
	       0,-1,0,
	       0,0,1,
	       0,0,-1}; 

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



static int* _select_neighborhood_system(int ngb_size) {
  if (ngb_size == 6) { 
    fprintf(stderr, "6-neighborhood system\n"); 
    return ngb6;
  }
  else if (ngb_size == 26) {
    fprintf(stderr, "26-neighborhood system\n"); 
    return ngb26;
  }
  else {
    fprintf(stderr, "Unknown neighborhood system\n");
    return NULL; 
  }
}



/*
  Perform the VE-step of a VEM algorithm for a general Markov random
  field segmentation model.

  ppm assumed C-contiguous double (X, Y, Z, K) 
  ref assumed C-contiguous double (NPTS, K)
  XYZ assumed C-contiguous unsigned int (NPTS, 3)
*/

#define TINY 1e-300


static void _ngb_integrate(double* res,
			   const PyArrayObject* ppm,
			   int x,
			   int y, 
			   int z,
			   const double* U, 
			   double beta, 
			   const int* ngb,
			   int ngb_size)			
{
  int j = 0, xn, yn, zn, k, kk, K = ppm->dimensions[3]; 
  const int* buf_ngb; 
  const double* ppm_data = (double*)ppm->data; 
  double *buf, *buf_ppm, *q, *buf_U;
  unsigned int u3 = K; 
  unsigned int u2 = ppm->dimensions[2]*u3; 
  unsigned int u1 = ppm->dimensions[1]*u2;
  unsigned int posmax = ppm->dimensions[0]*u1 - K;
  long int pos; 

  /*  Re-initialize output array */
  memset ((void*)res, 0, K*sizeof(double));

  /* Loop over neighbors */
  buf_ngb = ngb; 
  while (j < ngb_size) {
    xn = x + *buf_ngb; buf_ngb++; 
    yn = y + *buf_ngb; buf_ngb++;
    zn = z + *buf_ngb; buf_ngb++;
    pos = xn*u1 + yn*u2 + zn*u3;

    /* Ignore neighbor if outside the grid boundaries */
    if ((pos < 0) || (pos > posmax))
      continue; 

    /* Compute U*q */
    buf_ppm = (double*)ppm_data + pos;
    for (k=0, buf=res, buf_U=(double*)U; k<K; k++, buf++)
      for (kk=0, q=buf_ppm; kk<K; kk++, q++, buf_U++)
	*buf += *buf_U * *q;

    j ++; 
  }

  /* Finalize total message computation */
  for (k=0, buf=res; k<K; k++, buf++) 
    *buf = exp(-beta * (*buf));

  return; 
}


void ve_step(PyArrayObject* ppm, 
	     const PyArrayObject* ref,
	     const PyArrayObject* XYZ, 
	     const PyArrayObject* U,
	     int ngb_size,
	     double beta)

{
  int k, K, kk, x, y, z;
  double *p, *buf;
  double psum, tmp;  
  PyArrayIterObject* iter;
  int axis = 1; 
  double* ppm_data;
  unsigned int u3 = ppm->dimensions[3]; 
  unsigned int u2 = ppm->dimensions[2]*u3; 
  unsigned int u1 = ppm->dimensions[1]*u2;
  const double* ref_data = (double*)ref->data;
  const double* U_data = (double*)U->data;
  unsigned int v1 = ref->dimensions[1];
  unsigned int* xyz;
  int* ngb;

  /* Neighborhood system */
  ngb = _select_neighborhood_system(ngb_size);

  /* Number of classes */
  K = PyArray_DIM((PyArrayObject*)ppm, 3);

  /* Pointer to the data array */
  ppm_data = (double*)ppm->data;
  
  /* Allocate auxiliary vectors */
  p = (double*)calloc(K, sizeof(double)); 

  /* Loop over points */ 
  iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)XYZ, &axis);

  while(iter->index < iter->size) {

    /* Compute the average ppm in the neighborhood */
    xyz = PyArray_ITER_DATA(iter);
    x = xyz[0];
    y = xyz[1];
    z = xyz[2];
    _ngb_integrate(p, ppm, x, y, z, U_data, beta, (const int*)ngb, ngb_size);
    
    /* Multiply with reference and compute normalization constant */
    psum = 0.0;
    for (k=0, kk=(iter->index)*v1, buf=p; k<K; k++, kk++, buf++) {
      tmp = (*buf) * ref_data[kk];
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
	ppm_data[kk] = (*buf+TINY/(double)K)/(psum+TINY); 

    /* Update iterator */ 
    PyArray_ITER_NEXT(iter); 
  
  }

  /* Free memory */ 
  free(p);
  Py_XDECREF(iter);

  return; 
}


/* 
   Given an array of indices representing points in a regular 3d grid,
   compute the list of index pairs corresponding to connected points
   according to a given neighborhood system.

   `idx` array assumed C-contiguous int, values are the indices of the
   voxels in the non-masked 3d array, negative for masked voxels.

   Returned array `edges` has shape (NEDGES, 2) where the number of
   edges is to be determined within the function.
*/

PyArrayObject* make_edges(const PyArrayObject* idx,
			  int ngb_size)
{
  int* ngb = _select_neighborhood_system(ngb_size); 
  PyArrayIterObject* iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)idx);
  int* buf_ngb; 
  unsigned int xi, yi, zi, xj, yj, zj;
  unsigned int u2 = idx->dimensions[2]; 
  unsigned int u1 = idx->dimensions[1]*u2;
  unsigned int u0 = idx->dimensions[0]*u1;
  unsigned int mask_size = 0, n_edges = 0;
  int idx_i;
  int *buf_idx;
  unsigned int *edges_data, *buf_edges;
  unsigned int j;
  long int pos;
  PyArrayObject* edges;
  npy_intp dim[2] = {0, 2};
 
  /* First loop over the input array to determine the mask size */
  while(iter->index < iter->size) {
    buf_idx = (int*)PyArray_ITER_DATA(iter);
    if (*buf_idx >= 0)
      mask_size ++;
    PyArray_ITER_NEXT(iter); 
  }

  /* Allocate the array of edges using an upper bound of the required
     memory space */
  edges_data = (unsigned int*)malloc(2 * ngb_size * mask_size * sizeof(unsigned int)); 

  /* Second loop over the input array */
  PyArray_ITER_RESET(iter);
  iter->contiguous = 0; /* To force coordinates to be updated */
  buf_edges = edges_data;
  while(iter->index < iter->size) {

    xi = iter->coordinates[0];
    yi = iter->coordinates[1]; 
    zi = iter->coordinates[2]; 
    buf_idx = (int*)PyArray_ITER_DATA(iter);
    idx_i = *buf_idx;

    /* Loop over neighbors if current point is within the mask */
    if (idx_i >= 0) {
      buf_ngb = ngb;
      j = 0;
      while (j < ngb_size) {

	/* Get neighbor coordinates */
	xj = xi + *buf_ngb; buf_ngb++; 
	yj = yi + *buf_ngb; buf_ngb++;
	zj = zi + *buf_ngb; buf_ngb++;
	pos = xj*u1 + yj*u2 + zj;
	j ++;

	/* Store edge if neighbor is within the mask */
	if ((pos < 0) || (pos >= u0))
	  continue;
	buf_idx = (int*)idx->data + pos;
	if (*buf_idx < 0)
	  continue;
	buf_edges[0] = idx_i;
	buf_edges[1] = *buf_idx;
	n_edges ++;
	buf_edges += 2;

      }
    }
    
    /* Increment iterator */
    PyArray_ITER_NEXT(iter); 
    
  }

  /* Reallocate edges array to account for connections suppressed due to masking */
  edges_data = realloc((void *)edges_data, 2 * n_edges * sizeof(unsigned int)); 
  dim[0] = n_edges;
  edges = (PyArrayObject*) PyArray_SimpleNewFromData(2, dim, NPY_UINT, (void*)edges_data);

  /* Free memory */
  Py_XDECREF(iter);

  return edges;
}
