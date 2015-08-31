#include "mrf.h"

#include <math.h>
#include <stdlib.h>

#ifdef _MSC_VER
#define inline __inline
#endif


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
  if (ngb_size == 6)
    return ngb6;
  else if (ngb_size == 26) 
    return ngb26;
  else {
    fprintf(stderr, "Unknown neighborhood system\n");
    return NULL; 
  }
}



/*
  Perform the VE-step of a VEM algorithm for a general Markov random
  field segmentation model.

  Compute exp[-2 * beta * SUM_j (U * qj)] for a given voxel, where the
  sum is on the neighbors.

  ppm assumed C-contiguous double (X, Y, Z, K) 
  ref assumed C-contiguous double (NPTS, K)
  XYZ assumed C-contiguous npy_intp (NPTS, 3)
*/

#define TINY 1e-300

/* Compute neighborhood 'agreement' term required by the VE-step at a
particular voxel */
static void _ngb_integrate(double* res,
			   const PyArrayObject* ppm,
			   npy_intp x,
			   npy_intp y, 
			   npy_intp z,
			   const double* U, 
			   const int* ngb,
			   npy_intp ngb_size)
{
  npy_intp xn, yn, zn, pos, ngb_idx, k, kk;
  const int* buf_ngb;
  const double* ppm_data = (double*)ppm->data; 
  double *buf, *buf_ppm, *q, *buf_U;
  npy_intp K = ppm->dimensions[3]; 
  npy_intp u2 = ppm->dimensions[2]*K; 
  npy_intp u1 = ppm->dimensions[1]*u2;
  npy_intp posmax = ppm->dimensions[0]*u1 - K;

  /*  Re-initialize output array */
  memset((void*)res, 0, K*sizeof(double));

  /* Loop over neighbors */
  buf_ngb = ngb; 
  for (ngb_idx=0; ngb_idx<ngb_size; ngb_idx++) {
    xn = x + *buf_ngb; buf_ngb++; 
    yn = y + *buf_ngb; buf_ngb++;
    zn = z + *buf_ngb; buf_ngb++;
    pos = xn*u1 + yn*u2 + zn*K;

    /* Ignore neighbor if outside the grid boundaries */
    if ((pos < 0) || (pos > posmax))
      continue; 

    /* Compute U*q */
    buf_ppm = (double*)ppm_data + pos;
    for (k=0, buf=res, buf_U=(double*)U; k<K; k++, buf++)
      for (kk=0, q=buf_ppm; kk<K; kk++, q++, buf_U++)
	*buf += *buf_U * *q;
  }

  return; 
}


void ve_step(PyArrayObject* ppm, 
	     const PyArrayObject* ref,
	     const PyArrayObject* XYZ, 
	     const PyArrayObject* U,
	     int ngb_size,
	     double beta)

{
  npy_intp k, x, y, z, pos;
  double *p, *buf, *ppm_data;
  double psum, tmp;  
  PyArrayIterObject* iter;
  int axis = 1; 
  npy_intp K = ppm->dimensions[3]; 
  npy_intp u2 = ppm->dimensions[2]*K; 
  npy_intp u1 = ppm->dimensions[1]*u2;
  const double* ref_data = (double*)ref->data;
  const double* U_data = (double*)U->data;
  npy_intp* xyz;
  int* ngb;

  /* Neighborhood system */
  ngb = _select_neighborhood_system(ngb_size);

  /* Pointer to the data array */
  ppm_data = (double*)ppm->data;
  
  /* Allocate auxiliary vectors */
  p = (double*)calloc(K, sizeof(double)); 

  /* Loop over points */ 
  iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)XYZ, &axis);

  while(iter->index < iter->size) {

    /* Integrate the energy over the neighborhood */
    xyz = PyArray_ITER_DATA(iter);
    x = xyz[0];
    y = xyz[1];
    z = xyz[2];
    _ngb_integrate(p, ppm, x, y, z, U_data, (const int*)ngb, ngb_size);

    /* Apply exponential transform, multiply with reference and
       compute normalization constant */
    psum = 0.0;
    for (k=0, pos=(iter->index)*K, buf=p; k<K; k++, pos++, buf++) {
      tmp = exp(-2 * beta * (*buf)) * ref_data[pos];
      psum += tmp;
      *buf = tmp;
    }
    
    /* Normalize to unitary sum */
    pos = x*u1 + y*u2 + z*K; 
    if (psum > TINY) 
      for (k=0, buf=p; k<K; k++, pos++, buf++)
	ppm_data[pos] = *buf/psum; 
    else
      for (k=0, buf=p; k<K; k++, pos++, buf++)
	ppm_data[pos] = (*buf+TINY/(double)K)/(psum+TINY); 

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

   `idx` array assumed C-contiguous npy_intp, values are the indices
   of the voxels in the non-masked 3d array, negative for masked
   voxels.

   Returned array `edges` has shape (NEDGES, 2) where the number of
   edges is to be determined within the function.
*/

PyArrayObject* make_edges(const PyArrayObject* idx,
			  int ngb_size)
{
  int* ngb = _select_neighborhood_system(ngb_size);
  PyArrayIterObject* iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)idx);
  int* buf_ngb; 
  npy_intp xi, yi, zi, xj, yj, zj;
  npy_intp u2 = idx->dimensions[2]; 
  npy_intp u1 = idx->dimensions[1]*u2;
  npy_intp u0 = idx->dimensions[0]*u1;
  npy_intp mask_size = 0, n_edges = 0;
  npy_intp idx_i;
  npy_intp *buf_idx;
  npy_intp *edges_data, *buf_edges;
  npy_intp ngb_idx;
  npy_intp pos;
  PyArrayObject* edges;
  npy_intp dim[2] = {0, 2};
 
  /* First loop over the input array to determine the mask size */
  while(iter->index < iter->size) {
    buf_idx = (npy_intp*)PyArray_ITER_DATA(iter);
    if (*buf_idx >= 0)
      mask_size ++;
    PyArray_ITER_NEXT(iter); 
  }

  /* Allocate the array of edges using an upper bound of the required
     memory space */
  edges_data = (npy_intp*)malloc(2 * ngb_size * mask_size * sizeof(npy_intp)); 

  /* Second loop over the input array */
  PyArray_ITER_RESET(iter);
  iter->contiguous = 0; /* To force coordinates to be updated */
  buf_edges = edges_data;
  while(iter->index < iter->size) {

    xi = iter->coordinates[0];
    yi = iter->coordinates[1]; 
    zi = iter->coordinates[2]; 
    buf_idx = (npy_intp*)PyArray_ITER_DATA(iter);
    idx_i = *buf_idx;

    /* Loop over neighbors if current point is within the mask */
    if (idx_i >= 0) {
      buf_ngb = ngb;
      for (ngb_idx=0; ngb_idx<ngb_size; ngb_idx++) {

	/* Get neighbor coordinates */
	xj = xi + *buf_ngb; buf_ngb++; 
	yj = yi + *buf_ngb; buf_ngb++;
	zj = zi + *buf_ngb; buf_ngb++;
	pos = xj*u1 + yj*u2 + zj;

	/* Store edge if neighbor is within the mask */
	if ((pos < 0) || (pos >= u0))
	  continue;
	buf_idx = (npy_intp*)idx->data + pos;
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
  edges_data = realloc((void *)edges_data, 2 * n_edges * sizeof(npy_intp)); 
  dim[0] = n_edges;
  edges = (PyArrayObject*) PyArray_SimpleNewFromData(2, dim, NPY_INTP, (void*)edges_data);

  /* Transfer ownership to python (to avoid memory leaks!) */
  edges->flags = (edges->flags) | NPY_OWNDATA;

  /* Free memory */
  Py_XDECREF(iter);

  return edges;
}


/* 
   Compute the interaction energy:

   sum_i,j qi^T U qj
   = sum_i qi^T sum_j U qj   
   
*/
double interaction_energy(PyArrayObject* ppm, 
			  const PyArrayObject* XYZ,
			  const PyArrayObject* U,
			  int ngb_size)

{
  npy_intp k, x, y, z, pos;
  double *p, *buf;
  double res = 0.0, tmp;  
  PyArrayIterObject* iter;
  int axis = 1; 
  double* ppm_data;
  npy_intp K = ppm->dimensions[3]; 
  npy_intp u2 = ppm->dimensions[2]*K; 
  npy_intp u1 = ppm->dimensions[1]*u2;
  npy_intp* xyz; 
  const double* U_data = (double*)U->data;
  int* ngb;

  /* Neighborhood system */
  ngb = _select_neighborhood_system(ngb_size);

  /* Pointer to ppm array */
  ppm_data = (double*)ppm->data;

  /* Allocate auxiliary vector */
  p = (double*)calloc(K, sizeof(double)); 
  
  /* Loop over points */ 
  iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)XYZ, &axis);
  while(iter->index < iter->size) {
    
    /* Compute the average ppm in the neighborhood */ 
    xyz = PyArray_ITER_DATA(iter); 
    x = xyz[0];
    y = xyz[1];
    z = xyz[2];
    _ngb_integrate(p, ppm, x, y, z, U_data, (const int*)ngb, ngb_size);
    
    /* Calculate the dot product qi^T p where qi is the local
       posterior */
    tmp = 0.0; 
    pos = x*u1 + y*u2 + z*K; 
    for (k=0, buf=p; k<K; k++, pos++, buf++)
      tmp += ppm_data[pos]*(*buf);

    /* Update overall energy */ 
    res += tmp; 

    /* Update iterator */ 
    PyArray_ITER_NEXT(iter); 
  }

  /* Free memory */ 
  free(p);
  Py_XDECREF(iter);

  return res; 
}
