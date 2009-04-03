#include "cubic_spline.h"

#include <randomkit.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define ABS(a) ( (a) > 0.0 ? (a) : (-(a)) )

#define CUBIC_SPLINE_MIRROR(x, n, p) \
  ((x)<0.0 ? (-(x)) : ((x)>(n) ? ((p)-(x)) : (x)))




static void _cubic_spline_transform1d(double* res, double* src, unsigned int dim, 
				      unsigned int res_stride, unsigned int src_stride); 
static void _cubic_spline_transform(PyArrayObject* res, int axis, double* work); 
static inline void _copy_double_buffer(double* res, double* src, unsigned int dim, unsigned int src_stride);






/* Numpy import */
void cubic_spline_import_array(void) { 
  import_array(); 
  return;
}


/* Returns the value of the cubic B-spline function at x */
double cubic_spline_basis (double x)
{

  double y, absx, aux;

  absx = ABS(x);

  if (absx >= 2) 
    return 0.0;

  if (absx < 1) {
    aux = absx*absx;
    y = 0.66666666666667 - aux + 0.5*absx*aux;
  }  
  else {
    aux = 2 - absx;
    y = aux*aux*aux / 6.0;
  }
 
  return y;
}



/* 
   Assumes that src and res are same size and both point to DOUBLE buffers. 
*/

static void _cubic_spline_transform1d(double* res, double* src, unsigned int dim, 
				      unsigned int res_stride, unsigned int src_stride)
{
  int k; 
  double cp, cm, z1_k;
  double *buf_src, *buf_res;
  const double z1 = -0.26794919243112; /* -2 + sqrt(3) */
  const double cz1 = 0.28867513459481; /* z1/(z1^2-1) */

  /* 
     Initial value for the causal recursion.
     We use a mirror symmetric boundary condition for the discrete signal,
     yielding:
     
     cp(0) = (1/2-z1^(2N-2)) \sum_{k=0}^{2N-3} s(k) z1^k s(k),
     
     where we set: s(N)=s(N-2), s(N+1)=s(N-3), ..., s(2N-3)=s(1).
  */
  buf_src = src;
  cp = *buf_src;
  z1_k = 1;
  for (k=1; k<dim; k++) {
    z1_k = z1 * z1_k;   /* == z1^k */
    buf_src += src_stride;            /* points towards s[k] */
    cp += (*buf_src) * z1_k;
  }

  /* At this point, we have: z1_k = z1^(N-1) */
  for (k=2; k<dim; k++) {
    z1_k = z1 * z1_k;  
    buf_src -= src_stride;
    cp += (*buf_src) * z1_k;
  }
  
  /* At this point, we have: z1_k = z1^(2N-3) */
  z1_k = z1 * z1_k;
  cp = cp / (1 - z1_k);


  /* Storing the first causal coefficient */ 
  buf_res = res;
  *buf_res = cp;

  /* Do the causal recursion : [0..N-2]*/
  buf_src = src;
  for (k=1; k<dim; k++) {
    buf_src += src_stride; 
    cp = *buf_src + z1 * cp;
    buf_res += res_stride;
    *buf_res = cp;
  }

  /* Initial value for the anticausal recursion */
  cm = cz1 * (2.0 * cp - *buf_src);
  *buf_res = 6.0 * cm;
 
  /* Do the anti causal recursion : [N-2..0] */
  /* for (k=(dim-2); ((int)k)>=0; k--) { */
  for (k=1; k<dim; k++) {
    buf_res -= res_stride;     /* buf_res points towards the k-th index */
    cm = z1 * (cm - *buf_res);
    *buf_res = 6.0 * cm;
  }

  return;
}



static inline void _copy_double_buffer(double* res, double* src, unsigned int dim, unsigned int src_stride)
{
  unsigned int i; 
  double *buf_res=res, *buf_src=src; 

  for (i=0; i<dim; i++, buf_res++, buf_src+=src_stride)
    *buf_res = *buf_src; 

  return; 
}

/*
  res array must be double. 
  res and src must be of the same size. 
  work needs be 1d C-style contiguous double with size (at least) equal to res in the axis direction
*/

static void _cubic_spline_transform(PyArrayObject* res, int axis, double* work)
{
  PyArrayIterObject* iter;
  unsigned int dim, stride;

  /* Instantiate iterator and views */ 
  iter = (PyArrayIterObject*)PyArray_IterAllButAxis((PyObject*)res, &axis);
  dim = PyArray_DIM((PyArrayObject*)iter->ao, axis); 
  stride = sizeof(double)*PyArray_STRIDE((PyArrayObject*)iter->ao, axis); 

  /* Apply the cubic spline transform along given axis */ 
  while(iter->index < iter->size) {
    _copy_double_buffer(work, PyArray_ITER_DATA(iter), dim, stride); 
    _cubic_spline_transform1d(PyArray_ITER_DATA(iter), work, dim, stride, 1); 
    PyArray_ITER_NEXT(iter); 
  }

  /* Free local structures */ 
  free(work); 
  Py_DECREF(iter); 

  return; 
}


void cubic_spline_transform(PyArrayObject* res, const PyArrayObject* src)
{
  double* work; 
  unsigned int axis, aux=0, dimmax=0; 
  
  /* Copy src into res */ 
  PyArray_CastTo(res, (PyArrayObject*)src); 

  /* Compute the maximum array dimension over axes */ 
  for(axis=0; axis<res->nd; axis++) {
    aux = PyArray_DIM(res, axis);
    if (aux > dimmax) 
      dimmax = aux; 
  }

  /* Allocate auxiliary buffer */ 
  work = malloc(sizeof(double)*dimmax); 

  /* Apply separable cubic spline transforms */ 
  for(axis=0; axis<res->nd; axis++) 
    _cubic_spline_transform(res, axis, work);

  /* Free auxiliary buffer */ 
  free(work); 

  return; 
}


/* 

Assumes: -(dimX-1) <= x <= 2*(dimX-1) 
and similarly for other coordinates. 

Returns 0 otherwise. 

*/

double cubic_spline_sample1d (double x, const PyArrayObject* Coef) 
{

  unsigned int dim = PyArray_DIM(Coef, 0); 
  unsigned int offset = sizeof(double)*PyArray_STRIDE(Coef, 0); 
  double *coef = PyArray_DATA(Coef); 

  const unsigned int ddim = dim-1;
  const unsigned int two_ddim = 2*ddim;

  double *buf;
  int nx, px, xx;
  double s, aux;
  double bspx[4];
  int posx[4];
  double *buf_bspx;
  int *buf_posx;

  /* Right up superior point */
  aux = x + ddim; 
  if ((aux<0) || (aux>3*ddim)) 
    return 0.0;
  px = (int)(aux+2) - ddim;

  /* Left down inferior point */
  nx = px - 3;
  
  /* Compute the B-spline values as well as the image positions 
     where to find the B-spline coefficients (including mirror conditions) */ 
  buf_bspx = (double*)bspx;
  buf_posx = (int*)posx;
  for (xx = nx; xx <= px; xx ++, buf_bspx ++, buf_posx ++) {
    *buf_bspx = cubic_spline_basis(x-(double)xx);
    *buf_posx = CUBIC_SPLINE_MIRROR(xx, ddim, two_ddim);
  }

  /* Compute the interpolated value incrementally */
  s = 0.0;
  buf_bspx = (double*)bspx;
  buf_posx = (int*)posx;
  
  for (xx = nx; xx <= px; xx ++, buf_bspx ++, buf_posx ++) {
    
    /* Point towards the coefficient value at position xx */
    buf = coef + (*buf_posx)*offset;
    
    /* Update signal value */
    s += (*buf) * (*buf_bspx);
    
  }
    
  return s;
  
}




double cubic_spline_sample2d (double x, double y, const PyArrayObject* Coef)
{

  unsigned int dimX = PyArray_DIM(Coef, 0);
  unsigned int dimY = PyArray_DIM(Coef, 1);
  unsigned int offX = sizeof(double)*PyArray_STRIDE(Coef, 0); 
  unsigned int offY = sizeof(double)*PyArray_STRIDE(Coef, 1); 
  double *coef = PyArray_DATA(Coef); 

  const unsigned int ddimX = dimX-1;
  const unsigned int ddimY = dimY-1;
  const unsigned int two_ddimX = 2*ddimX;
  const unsigned int two_ddimY = 2*ddimY;

  double *buf;
  int nx, ny, px, py, xx, yy;
  double s, aux;
  double bspx[4], bspy[4];
  int posx[4], posy[4];
  double *buf_bspx, *buf_bspy;
  int *buf_posx, *buf_posy;
  int shfty;


  /* Right up superior point */
  aux = x + ddimX; 
  if ((aux<0) || (aux>3*ddimX)) 
    return 0.0;
  px = (int)(aux+2) - ddimX;

  aux = y + ddimY; 
  if ((aux<0) || (aux>3*ddimY)) 
    return 0.0;
  py = (int)(aux+2) - ddimY;


  /* Left down inferior point */
  nx = px - 3;
  ny = py - 3;

  /* Compute the B-spline values as well as the image positions 
     where to find the B-spline coefficients (including mirror conditions) */ 
  buf_bspx = (double*)bspx;
  buf_posx = (int*)posx;
  for (xx = nx; xx <= px; xx ++, buf_bspx ++, buf_posx ++) {
    *buf_bspx = cubic_spline_basis(x-(double)xx);
    *buf_posx = CUBIC_SPLINE_MIRROR(xx, ddimX, two_ddimX);
  }

  buf_bspy = (double*)bspy;
  buf_posy = (int*)posy;
  for (yy = ny; yy <= py; yy ++, buf_bspy ++, buf_posy ++) {
    *buf_bspy = cubic_spline_basis(y-(double)yy);
    *buf_posy = CUBIC_SPLINE_MIRROR(yy, ddimY, two_ddimY);
  }


  /* Compute the interpolated value incrementally */
  s = 0.0;
  buf_bspy = (double*)bspy;
  buf_posy = (int*)posy;
    
  for (yy = ny; yy <= py; yy ++, buf_bspy ++, buf_posy ++) {
    
    aux = 0.0;
    buf_bspx = (double*)bspx;
    buf_posx = (int*)posx;
    shfty = offY*(*buf_posy);

    for (xx = nx; xx <= px; xx ++, buf_bspx ++, buf_posx ++) {
    
      /* Point towards the coefficient value at position (xx, yy, zz) */
      buf = coef + offX*(*buf_posx) + shfty;

      /* Update signal value */
      aux += (*buf) * (*buf_bspx);
    
    }
    
    s += aux * (*buf_bspy); 

  }
  
  return s;

}


double cubic_spline_sample3d (double x, double y, double z, const PyArrayObject* Coef)
{
  unsigned int dimX = PyArray_DIM(Coef, 0);
  unsigned int dimY = PyArray_DIM(Coef, 1);
  unsigned int dimZ = PyArray_DIM(Coef, 2);
  unsigned int offX = sizeof(double)*PyArray_STRIDE(Coef, 0); 
  unsigned int offY = sizeof(double)*PyArray_STRIDE(Coef, 1); 
  unsigned int offZ = sizeof(double)*PyArray_STRIDE(Coef, 2); 
  double *coef = PyArray_DATA(Coef); 

  const unsigned int ddimX = dimX-1;
  const unsigned int ddimY = dimY-1;
  const unsigned int ddimZ = dimZ-1;
  const unsigned int two_ddimX = 2*ddimX;
  const unsigned int two_ddimY = 2*ddimY;
  const unsigned int two_ddimZ = 2*ddimZ;

  double *buf;
  int nx, ny, nz, px, py, pz;
  int xx, yy, zz;
  double s, aux, aux2;
  double bspx[4], bspy[4], bspz[4]; 
  int posx[4], posy[4], posz[4];
  double *buf_bspx, *buf_bspy, *buf_bspz;
  int *buf_posx, *buf_posy, *buf_posz;
  int shftyz, shftz;

  /* Right up superior point */
  aux = x + ddimX; 
  if ((aux<0) || (aux>3*ddimX)) 
    return 0.0;
  px = (int)(aux+2) - ddimX;

  aux = y + ddimY; 
  if ((aux<0) || (aux>3*ddimY)) 
    return 0.0;
  py = (int)(aux+2) - ddimY;

  aux = z + ddimZ; 
  if ((aux<0) || (aux>3*ddimZ)) 
    return 0.0;
  pz = (int)(aux+2) - ddimZ;

  /* Left down inferior point */
  nx = px - 3;
  ny = py - 3;
  nz = pz - 3;
  
  /* Compute the B-spline values as well as the image positions 
     where to find the B-spline coefficients (including mirror conditions) */ 
  buf_bspx = (double*)bspx;
  buf_posx = (int*)posx;
  for (xx = nx; xx <= px; xx ++, buf_bspx ++, buf_posx ++) {
    *buf_bspx = cubic_spline_basis(x-(double)xx);
    *buf_posx = CUBIC_SPLINE_MIRROR(xx, ddimX, two_ddimX);
  }

  buf_bspy = (double*)bspy;
  buf_posy = (int*)posy;
  for (yy = ny; yy <= py; yy ++, buf_bspy ++, buf_posy ++) {
    *buf_bspy = cubic_spline_basis(y-(double)yy);
    *buf_posy = CUBIC_SPLINE_MIRROR(yy, ddimY, two_ddimY);
  }

  buf_bspz = (double*)bspz;
  buf_posz = (int*)posz;
  for (zz = nz; zz <= pz; zz ++, buf_bspz ++, buf_posz ++) {
    *buf_bspz = cubic_spline_basis(z-(double)zz);
    *buf_posz = CUBIC_SPLINE_MIRROR(zz, ddimZ, two_ddimZ);
  }

  /* Compute the interpolated value incrementally */
  s = 0.0;
  buf_bspz = (double*)bspz;
  buf_posz = (int*)posz;

  for (zz = nz; zz <= pz; zz ++, buf_bspz ++, buf_posz ++) {

    aux2 = 0.0;
    buf_bspy = (double*)bspy;
    buf_posy = (int*)posy;
    shftz = offZ*(*buf_posz);
    
    for (yy = ny; yy <= py; yy ++, buf_bspy ++, buf_posy ++) {
      
      aux = 0.0;
      buf_bspx = (double*)bspx;
      buf_posx = (int*)posx;
      shftyz = offY*(*buf_posy) + shftz;

      for (xx = nx; xx <= px; xx ++, buf_bspx ++, buf_posx ++) {
	
	/* Point towards the coefficient value at position (xx, yy, zz) */
	buf = coef + offX*(*buf_posx) + shftyz;
	
	/* Update signal value */
	aux += (*buf) * (*buf_bspx);
	
      } /* end loop on x */
      aux2 += aux * (*buf_bspy); 

    }  /* end loop on y */
    s += aux2 * (*buf_bspz); 
    
  } /* end loop on z */
  

  return s;

}



double cubic_spline_sample4d (double x, double y, double z, double t, const PyArrayObject* Coef)
{
  unsigned int dimX = PyArray_DIM(Coef, 0);
  unsigned int dimY = PyArray_DIM(Coef, 1);
  unsigned int dimZ = PyArray_DIM(Coef, 2);
  unsigned int dimT = PyArray_DIM(Coef, 3);
  unsigned int offX = sizeof(double)*PyArray_STRIDE(Coef, 0); 
  unsigned int offY = sizeof(double)*PyArray_STRIDE(Coef, 1); 
  unsigned int offZ = sizeof(double)*PyArray_STRIDE(Coef, 2); 
  unsigned int offT = sizeof(double)*PyArray_STRIDE(Coef, 3); 
  double *coef = PyArray_DATA(Coef); 

  const unsigned int ddimX = dimX-1;
  const unsigned int ddimY = dimY-1;
  const unsigned int ddimZ = dimZ-1;
  const unsigned int ddimT = dimT-1;
  const unsigned int two_ddimX = 2*ddimX;
  const unsigned int two_ddimY = 2*ddimY;
  const unsigned int two_ddimZ = 2*ddimZ;
  const unsigned int two_ddimT = 2*ddimT;

  double *buf;
  int nx, ny, nz, nt, px, py, pz, pt;
  int xx, yy, zz, tt;
  double s, aux, aux2, aux3;
  double bspx[4], bspy[4], bspz[4], bspt[4]; 
  int posx[4], posy[4], posz[4], post[4];
  double *buf_bspx, *buf_bspy, *buf_bspz, *buf_bspt;
  int *buf_posx, *buf_posy, *buf_posz, *buf_post;
  int shftyzt, shftzt, shftt;


  /* Right up superior point */
  aux = x + ddimX; 
  if ((aux<0) || (aux>3*ddimX)) 
    return 0.0;
  px = (int)(aux+2) - ddimX;

  aux = y + ddimY; 
  if ((aux<0) || (aux>3*ddimY)) 
    return 0.0;
  py = (int)(aux+2) - ddimY;

  aux = z + ddimZ; 
  if ((aux<0) || (aux>3*ddimZ)) 
    return 0.0;
  pz = (int)(aux+2) - ddimZ;

  aux = t + ddimT; 
  if ((aux<0) || (aux>3*ddimT)) 
    return 0.0;
  pt = (int)(aux+2) - ddimT;

  /* Left down inferior point */
  nx = px - 3;
  ny = py - 3;
  nz = pz - 3;
  nt = pt - 3;
  
  /* Compute the B-spline values as well as the image positions 
     where to find the B-spline coefficients (including mirror conditions) */ 
  buf_bspx = (double*)bspx;
  buf_posx = (int*)posx;
  for (xx = nx; xx <= px; xx ++, buf_bspx ++, buf_posx ++) {
    *buf_bspx = cubic_spline_basis(x-(double)xx);
    *buf_posx = CUBIC_SPLINE_MIRROR(xx, ddimX, two_ddimX);
  }

  buf_bspy = (double*)bspy;
  buf_posy = (int*)posy;
  for (yy = ny; yy <= py; yy ++, buf_bspy ++, buf_posy ++) {
    *buf_bspy = cubic_spline_basis(y-(double)yy);
    *buf_posy = CUBIC_SPLINE_MIRROR(yy, ddimY, two_ddimY);
  }

  buf_bspz = (double*)bspz;
  buf_posz = (int*)posz;
  for (zz = nz; zz <= pz; zz ++, buf_bspz ++, buf_posz ++) {
    *buf_bspz = cubic_spline_basis(z-(double)zz);
    *buf_posz = CUBIC_SPLINE_MIRROR(zz, ddimZ, two_ddimZ);
  }

  buf_bspt = (double*)bspt;
  buf_post = (int*)post;
  for (tt = nt; tt <= pt; tt ++, buf_bspt ++, buf_post ++) {
    *buf_bspt = cubic_spline_basis(t-(double)tt);
    *buf_post = CUBIC_SPLINE_MIRROR(tt, ddimT, two_ddimT);
  }
  

  /* Compute the interpolated value incrementally by visiting the neighbors in turn  */
  s = 0.0;
  buf_bspt = (double*)bspt;
  buf_post = (int*)post;
  
  for (tt = nt; tt <= pt; tt ++, buf_bspt ++, buf_post ++) {
    
    aux3 = 0.0;
    buf_bspz = (double*)bspz;
    buf_posz = (int*)posz;
    shftt = offT*(*buf_post);
    
    for (zz = nz; zz <= pz; zz ++, buf_bspz ++, buf_posz ++) {
      
      aux2 = 0.0;
      buf_bspy = (double*)bspy;
      buf_posy = (int*)posy;
      shftzt =  offZ*(*buf_posz) + shftt;
      
      for (yy = ny; yy <= py; yy ++, buf_bspy ++, buf_posy ++) {
	
	aux = 0.0;
	buf_bspx = (double*)bspx;
	buf_posx = (int*)posx;
	shftyzt = offY*(*buf_posy) + shftzt;

	for (xx = nx; xx <= px; xx ++, buf_bspx ++, buf_posx ++) {
	  
	  /* Point towards the coefficient value at position (xx, yy, zz, tt) */
	  buf = coef + offX*(*buf_posx) + shftyzt;
	  
	  /* Update signal value */
	  aux += (*buf) * (*buf_bspx);
	  
	} /* end loop on x */
	aux2 += aux * (*buf_bspy); 
	
      }  /* end loop on y */
      aux3 += aux2 * (*buf_bspz); 
      
    } /* end loop on z */
    s += aux3 * (*buf_bspt); 
    
  } /* end loop on t */
  
  return s;

}


