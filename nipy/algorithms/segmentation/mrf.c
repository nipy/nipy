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


/* Compute the (negated) expected interaction energy of a voxel with
   some neighbor */
static inline void _get_message_mf(double* res, int K, size_t pos, 
				   const double* ppm_data, const double* aux)
{
  double *buf = res, *buf_ppm = (double*)ppm_data + pos;
  int k;
  
  for (k=0, buf, buf_ppm; k<K; k++, buf++, buf_ppm++)
    *buf += *buf_ppm;

  return;
}

static inline void _finalize_inbox_mf(double* res, int K, const double* aux) 
{
  int k; 
  double* buf;
  double aux0 = aux[0];

  for (k=0, buf=res; k<K; k++, buf++) 
    *buf = exp(aux0 * (*buf));

  return; 
}

static inline void _initialize_inbox_mf(double* res, int K)
{ 
  memset ((void*)res, 0, K*sizeof(double));
  return; 
}

static inline void _get_message_icm(double* res, int K, size_t pos,  
				    const double* ppm_data, const double* aux)
{
  int k, kmax = -1;
  double max = 0, tmp;
  double *buf_ppm = (double*)ppm_data + pos;

  for (k=0; k<K; k++, buf_ppm++) {
    tmp = *buf_ppm;
    if (tmp>max) {
      kmax = k;
      max = tmp;
    }
  }
  if (kmax >= 0)
    res[kmax] += 1;

  return;
}

static inline void _get_message_bp(double* res, int K, size_t pos, 
				   const double* ppm_data, const double* aux)
{
  double *buf = res, *buf_ppm = (double*)ppm_data + pos;
  int k;
  double aux0 = aux[0]; 

  for (k=0; k<K; k++, buf++, buf_ppm++) 
    *buf *= aux0 * (*buf_ppm) + 1; 
  
  return;
}

static inline void _initialize_inbox_bp(double* res, int K)
{ 
  double *buf = res; 
  int k; 

  for (k=0; k<K; k++, buf++)
    *buf = 1.0; 

  /* memset ((void*)res, 1, K*sizeof(double));*/
  return; 
}



/*
  Compute the incoming messages at a given voxel as a function of the
  class label, and aggregate them across neighbors.
  
  The vode_fn argument is a pointer to the function that actually
  computes the expected interaction energy with a particular neighbor.

  ppm assumed contiguous double (X, Y, Z, K) 
  res assumed preallocated with size >= K 

*/

static void _ngb26_compound_messages(double* res, 
				     const PyArrayObject* ppm,  
				     int x,
				     int y, 
				     int z,
				     void* initialize_inbox,
				     void* get_message,
				     void* finalize_inbox,
				     const double* aux)			
{
  int j = 0, xn, yn, zn, nn = 26, K = ppm->dimensions[3]; 
  int* buf_ngb; 
  const double* ppm_data = (double*)ppm->data; 
  size_t u3 = K; 
  size_t u2 = ppm->dimensions[2]*u3; 
  size_t u1 = ppm->dimensions[1]*u2;
  size_t pos; 
  void (*_initialize_inbox)(double*,int) = initialize_inbox;
  void (*_get_message)(double*,int,size_t,const double*,const double*) = get_message;
  void (*_finalize_inbox)(double*,int,const double*) = finalize_inbox;

  /*  Re-initialize output array */
  _initialize_inbox(res, K); 

  /* Loop over neighbors */
  buf_ngb = ngb26; 
  while (j < nn) {
    xn = x + *buf_ngb; buf_ngb++; 
    yn = y + *buf_ngb; buf_ngb++;
    zn = z + *buf_ngb; buf_ngb++;
    pos = xn*u1 + yn*u2 + zn*u3;
    _get_message(res, K, pos, ppm_data, aux);
    j ++; 
  }

  /* Finalize total message computation */
  if (_finalize_inbox != NULL) 
    _finalize_inbox(res, K, aux); 
  
  return; 
}




/*
  ppm assumed contiguous double (X, Y, Z, K) 
  ref assumed contiguous double (NPTS, K)
  XYZ assumed contiguous usigned int (NPTS, 3)
*/

#define TINY 1e-300

void ve_step(PyArrayObject* ppm, 
	     const PyArrayObject* ref,
	     const PyArrayObject* XYZ, 
	     double beta,
	     int copy,
	     int scheme)

{
  int k, K, kk, x, y, z;
  double *p, *buf;
  double psum, tmp;  
  PyArrayIterObject* iter;
  int axis = 1; 
  double* ppm_data;
  size_t u3 = ppm->dimensions[3]; 
  size_t u2 = ppm->dimensions[2]*u3; 
  size_t u1 = ppm->dimensions[1]*u2;
  const double* ref_data = (double*)ref->data;
  size_t v1 = ref->dimensions[1];
  int* xyz; 
  void (*initialize_inbox)(double*,int);
  void (*get_message)(double*,int,size_t,const double*,const double*);
  void (*finalize_inbox)(double*,int,const double*);
  size_t S; 
  double* aux;

  /* Dimensions */
  K = PyArray_DIM((PyArrayObject*)ppm, 3);
  S = PyArray_SIZE(ppm);

  /* Copy or not copy */
  if (copy) {
    ppm_data = (double*)calloc(S, sizeof(double));
    if (ppm_data==NULL) {
      fprintf(stderr, "Cannot allocate ppm copy\n"); 
      return; 
    }
    memcpy((void*)ppm_data, (void*)ppm->data, S*sizeof(double));
  }
  else
    ppm_data = (double*)ppm->data;
  
  /* Select message passging scheme: mean-field, ICM or
     belief-propagation */
  switch (scheme) {
  case 0: 
    {
      initialize_inbox = &_initialize_inbox_mf;
      get_message = &_get_message_mf;
      finalize_inbox = &_finalize_inbox_mf;
      aux = (double*)calloc(1, sizeof(double));   
      aux[0] = beta; 
    }
    break; 
  case 1: 
    {
      initialize_inbox = &_initialize_inbox_mf;
      get_message = &_get_message_icm;
      finalize_inbox = &_finalize_inbox_mf;
      aux = (double*)calloc(1, sizeof(double));
      aux[0] = beta; 
    }
    break; 
  case 2: 
    {
      initialize_inbox = &_initialize_inbox_bp;
      get_message = &_get_message_bp;    
      finalize_inbox = NULL; 
      aux = (double*)calloc(1, sizeof(double));
      aux[0] = exp(beta) - 1; 
      if (aux[0] < 0) 
	aux[0] = 0; 
    }
    break; 
  default: 
    {
      return; 
    }
    break; 
  }

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
    _ngb26_compound_messages(p, ppm, x, y, z, 
			     (void*)initialize_inbox, 
			     (void*)get_message, 
			     (void*)finalize_inbox, 
			     aux); 
    
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

  /* If applicable, copy back the auxiliary ppm array into the input */ 
  if (copy) {    
    memcpy((void*)ppm->data, (void*)ppm_data, S*sizeof(double));
    free(ppm_data);
  }

  /* Free memory */ 
  free(p);
  if (aux != NULL) 
    free(aux); 
  Py_XDECREF(iter);

  return; 
}



double interaction_energy(PyArrayObject* ppm, 
			  const PyArrayObject* XYZ)

{
  int k, K, kk, x, y, z;
  double *p, *buf;
  double res = 0.0, tmp;  
  PyArrayIterObject* iter;
  int axis = 1; 
  double* ppm_data;
  size_t u3 = ppm->dimensions[3]; 
  size_t u2 = ppm->dimensions[2]*u3; 
  size_t u1 = ppm->dimensions[1]*u2;
  int* xyz; 

  /* Dimensions */
  K = PyArray_DIM((PyArrayObject*)ppm, 3);
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
    _ngb26_compound_messages(p, ppm, x, y, z, &_initialize_inbox_mf, 
			     &_get_message_mf, NULL, NULL); 
    
    /* Calculate the dot product <q,p> where q is the local
       posterior */
    tmp = 0.0; 
    kk = x*u1 + y*u2 + z*u3; 
    for (k=0, buf=p; k<K; k++, kk++, buf++)
      tmp += ppm_data[kk]*(*buf);

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
