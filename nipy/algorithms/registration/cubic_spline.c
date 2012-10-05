#include "cubic_spline.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Useful marcos */ 
#define ABS(a) ( (a) > 0.0 ? (a) : (-(a)) )
#define FLOOR(a)((a)>0.0 ? (int)(a):(((int)(a)-a)!= 0.0 ? (int)(a)-1 : (int)(a)))  
#define ROUND(a)(FLOOR(a+0.5))

#ifdef _MSC_VER
#define inline __inline
#endif


/*
  Three different boundary conditions are implemented:
      mode == 0 : 'zero'
      mode == 1: 'nearest'
      mode == 2: 'reflect'

  Depending on the mode, the input coordinate x is mirrored so as to
  fall within the image bounds [0..ddim] and a weight w is computed.
 */
#define APPLY_BOUNDARY_CONDITIONS(mode, x, w, ddim)		\
  if (!_apply_boundary_conditions(mode, ddim, &x, &w))		\
    return 0.0;

#define COMPUTE_NEIGHBORS(x, ddim, nx, px)		\
  if (!_mirror_grid_neighbors(x, ddim, &nx, &px))	\
    return 0.0; 

/* 
   The following marco forces numpy to consider a PyArrayIterObject
   non-contiguous. Otherwise, coordinates won't be updated - don't
   know whether this is a bug or not. 
*/
#define UPDATE_ITERATOR_COORDS(iter)		\
  iter->contiguous = 0;


static void _cubic_spline_transform1d(double* res, double* src, unsigned int dim, 
				      unsigned int res_stride, unsigned int src_stride); 
static void _cubic_spline_transform(PyArrayObject* res, int axis, double* work); 
static inline void _copy_double_buffer(double* res, double* src, unsigned int dim,
				       unsigned int src_stride);
static inline int _mirrored_position(int x, unsigned int ddim);
static inline int _apply_boundary_conditions(int mode, unsigned int ddim, 
					     double* x, double* w);
static inline int _mirror_grid_neighbors(double x, unsigned int ddim, int* nx, int* px);
static inline void _apply_affine_transform(double* Tx, double* Ty, double* Tz, 
					   const double* Tvox, 
					   size_t x, size_t y, size_t z); 


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
  double z1 = -0.26794919243112; /* -2 + sqrt(3) */
  double cz1 = 0.28867513459481; /* z1/(z1^2-1) */

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
  stride = PyArray_STRIDE((PyArrayObject*)iter->ao, axis)/sizeof(double); 

  /* Apply the cubic spline transform along given axis */ 
  while(iter->index < iter->size) {
    _copy_double_buffer(work, PyArray_ITER_DATA(iter), dim, stride); 
    _cubic_spline_transform1d(PyArray_ITER_DATA(iter), work, dim, stride, 1); 
    PyArray_ITER_NEXT(iter); 
  }

  /* Free local structures */ 
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
  work = (double*)malloc(sizeof(double)*dimmax); 

  /* Apply separable cubic spline transforms */ 
  for(axis=0; axis<res->nd; axis++) 
    _cubic_spline_transform(res, axis, work);

  /* Free auxiliary buffer */ 
  free(work); 

  return; 
}


double cubic_spline_sample1d (double x, const PyArrayObject* Coef, int mode) 
{

  unsigned int ddim = PyArray_DIM(Coef, 0) - 1; 
  unsigned int offset = PyArray_STRIDE(Coef, 0)/sizeof(double); 
  double *coef = PyArray_DATA(Coef); 
  double *buf;
  int nx, px, xx;
  double s;
  double bspx[4];
  int posx[4];
  double *buf_bspx;
  int *buf_posx;
  double w = 1; 

  APPLY_BOUNDARY_CONDITIONS(mode, x, w, ddim); 
  COMPUTE_NEIGHBORS(x, ddim, nx, px);  

  /* Compute the B-spline values as well as the image positions 
     where to find the B-spline coefficients (including mirror conditions) */ 
  buf_bspx = (double*)bspx;
  buf_posx = (int*)posx;
  for (xx = nx; xx <= px; xx ++, buf_bspx ++, buf_posx ++) {
    *buf_bspx = cubic_spline_basis(x-(double)xx);
    *buf_posx = _mirrored_position(xx, ddim);
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
    
  return w*s;
}




double cubic_spline_sample2d (double x, double y, const PyArrayObject* Coef,
			      int mode_x, int mode_y)
{

  unsigned int ddimX = PyArray_DIM(Coef, 0) - 1;
  unsigned int ddimY = PyArray_DIM(Coef, 1) - 1;
  unsigned int offX = PyArray_STRIDE(Coef, 0)/sizeof(double); 
  unsigned int offY = PyArray_STRIDE(Coef, 1)/sizeof(double); 
  double *coef = PyArray_DATA(Coef); 
  double *buf;
  int nx, ny, px, py, xx, yy;
  double s, aux;
  double bspx[4], bspy[4];
  int posx[4], posy[4];
  double *buf_bspx, *buf_bspy;
  int *buf_posx, *buf_posy;
  int shfty;
  double wx = 1, wy = 1; 

  APPLY_BOUNDARY_CONDITIONS(mode_x, x, wx, ddimX); 
  COMPUTE_NEIGHBORS(x, ddimX, nx, px); 
  APPLY_BOUNDARY_CONDITIONS(mode_y, y, wy, ddimY); 
  COMPUTE_NEIGHBORS(y, ddimY, ny, py);  

  /* Compute the B-spline values as well as the image positions 
     where to find the B-spline coefficients (including mirror conditions) */ 
  buf_bspx = (double*)bspx;
  buf_posx = (int*)posx;
  for (xx = nx; xx <= px; xx ++, buf_bspx ++, buf_posx ++) {
    *buf_bspx = cubic_spline_basis(x-(double)xx);
    *buf_posx = _mirrored_position(xx, ddimX);
  }

  buf_bspy = (double*)bspy;
  buf_posy = (int*)posy;
  for (yy = ny; yy <= py; yy ++, buf_bspy ++, buf_posy ++) {
    *buf_bspy = cubic_spline_basis(y-(double)yy);
    *buf_posy = _mirrored_position(yy, ddimY);
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
  
  return wx*wy*s;

}


double cubic_spline_sample3d (double x, double y, double z, const PyArrayObject* Coef,
			      int mode_x, int mode_y, int mode_z)
{
  unsigned int ddimX = PyArray_DIM(Coef, 0) - 1;
  unsigned int ddimY = PyArray_DIM(Coef, 1) - 1;
  unsigned int ddimZ = PyArray_DIM(Coef, 2) - 1;
  unsigned int offX = PyArray_STRIDE(Coef, 0)/sizeof(double); 
  unsigned int offY = PyArray_STRIDE(Coef, 1)/sizeof(double); 
  unsigned int offZ = PyArray_STRIDE(Coef, 2)/sizeof(double); 
  double *coef = PyArray_DATA(Coef); 
  double *buf;
  int nx, ny, nz, px, py, pz;
  int xx, yy, zz;
  double s, aux, aux2;
  double bspx[4], bspy[4], bspz[4]; 
  int posx[4], posy[4], posz[4];
  double *buf_bspx, *buf_bspy, *buf_bspz;
  int *buf_posx, *buf_posy, *buf_posz;
  int shftyz, shftz;
  double wx = 1, wy = 1, wz = 1; 

  APPLY_BOUNDARY_CONDITIONS(mode_x, x, wx, ddimX); 
  COMPUTE_NEIGHBORS(x, ddimX, nx, px); 
  APPLY_BOUNDARY_CONDITIONS(mode_y, y, wy, ddimY); 
  COMPUTE_NEIGHBORS(y, ddimY, ny, py);  
  APPLY_BOUNDARY_CONDITIONS(mode_z, z, wz, ddimZ); 
  COMPUTE_NEIGHBORS(z, ddimZ, nz, pz);  

  /* Compute the B-spline values as well as the image positions 
     where to find the B-spline coefficients (including mirror conditions) */ 
  buf_bspx = (double*)bspx;
  buf_posx = (int*)posx;
  for (xx = nx; xx <= px; xx ++, buf_bspx ++, buf_posx ++) {
    *buf_bspx = cubic_spline_basis(x-(double)xx);
    *buf_posx = _mirrored_position(xx, ddimX);
  }

  buf_bspy = (double*)bspy;
  buf_posy = (int*)posy;
  for (yy = ny; yy <= py; yy ++, buf_bspy ++, buf_posy ++) {
    *buf_bspy = cubic_spline_basis(y-(double)yy);
    *buf_posy = _mirrored_position(yy, ddimY);
  }

  buf_bspz = (double*)bspz;
  buf_posz = (int*)posz;
  for (zz = nz; zz <= pz; zz ++, buf_bspz ++, buf_posz ++) {
    *buf_bspz = cubic_spline_basis(z-(double)zz);
    *buf_posz = _mirrored_position(zz, ddimZ);
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


  return wx*wy*wz*s;

}



double cubic_spline_sample4d (double x, double y, double z, double t, const PyArrayObject* Coef,
			      int mode_x, int mode_y, int mode_z, int mode_t)
{
  unsigned int ddimX = PyArray_DIM(Coef, 0) - 1;
  unsigned int ddimY = PyArray_DIM(Coef, 1) - 1;
  unsigned int ddimZ = PyArray_DIM(Coef, 2) - 1;
  unsigned int ddimT = PyArray_DIM(Coef, 3) - 1;
  unsigned int offX = PyArray_STRIDE(Coef, 0)/sizeof(double); 
  unsigned int offY = PyArray_STRIDE(Coef, 1)/sizeof(double); 
  unsigned int offZ = PyArray_STRIDE(Coef, 2)/sizeof(double); 
  unsigned int offT = PyArray_STRIDE(Coef, 3)/sizeof(double); 
  double *coef = PyArray_DATA(Coef); 
  double *buf;
  int nx, ny, nz, nt, px, py, pz, pt;
  int xx, yy, zz, tt;
  double s, aux, aux2, aux3;
  double bspx[4], bspy[4], bspz[4], bspt[4]; 
  int posx[4], posy[4], posz[4], post[4];
  double *buf_bspx, *buf_bspy, *buf_bspz, *buf_bspt;
  int *buf_posx, *buf_posy, *buf_posz, *buf_post;
  int shftyzt, shftzt, shftt;
  double wx = 1, wy = 1, wz = 1, wt = 1; 

  APPLY_BOUNDARY_CONDITIONS(mode_x, x, wx, ddimX); 
  COMPUTE_NEIGHBORS(x, ddimX, nx, px); 
  APPLY_BOUNDARY_CONDITIONS(mode_y, y, wy, ddimY); 
  COMPUTE_NEIGHBORS(y, ddimY, ny, py);  
  APPLY_BOUNDARY_CONDITIONS(mode_z, z, wz, ddimZ); 
  COMPUTE_NEIGHBORS(z, ddimZ, nz, pz);    
  APPLY_BOUNDARY_CONDITIONS(mode_t, t, wt, ddimT); 
  COMPUTE_NEIGHBORS(t, ddimT, nt, pt); 

  /* Compute the B-spline values as well as the image positions 
     where to find the B-spline coefficients (including mirror conditions) */ 
  buf_bspx = (double*)bspx;
  buf_posx = (int*)posx;
  for (xx = nx; xx <= px; xx ++, buf_bspx ++, buf_posx ++) {
    *buf_bspx = cubic_spline_basis(x-(double)xx);
    *buf_posx = _mirrored_position(xx, ddimX);
  }

  buf_bspy = (double*)bspy;
  buf_posy = (int*)posy;
  for (yy = ny; yy <= py; yy ++, buf_bspy ++, buf_posy ++) {
    *buf_bspy = cubic_spline_basis(y-(double)yy);
    *buf_posy = _mirrored_position(yy, ddimY);
  }

  buf_bspz = (double*)bspz;
  buf_posz = (int*)posz;
  for (zz = nz; zz <= pz; zz ++, buf_bspz ++, buf_posz ++) {
    *buf_bspz = cubic_spline_basis(z-(double)zz);
    *buf_posz = _mirrored_position(zz, ddimZ);
  }

  buf_bspt = (double*)bspt;
  buf_post = (int*)post;
  for (tt = nt; tt <= pt; tt ++, buf_bspt ++, buf_post ++) {
    *buf_bspt = cubic_spline_basis(t-(double)tt);
    *buf_post = _mirrored_position(tt, ddimT);
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
  
  return wx*wy*wz*wt*s;

}


/* 
   Resample a 3d image submitted to an affine transformation.
   Tvox is the voxel transformation from the image to the destination grid.  
*/
void cubic_spline_resample3d(PyArrayObject* im_resampled, const PyArrayObject* im,   
			     const double* Tvox, int cast_integer, 
			     int mode_x, int mode_y, int mode_z)
{
  double i1;
  PyObject* py_i1;
  PyArrayObject* im_spline_coeff;
  PyArrayIterObject* imIter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)im_resampled); 
  unsigned int x, y, z;
  unsigned dimX = PyArray_DIM(im, 0);
  unsigned dimY = PyArray_DIM(im, 1);
  unsigned dimZ = PyArray_DIM(im, 2);
  npy_intp dims[3] = {dimX, dimY, dimZ}; 
  double Tx, Ty, Tz;

  /* Compute the spline coefficient image */
  im_spline_coeff = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_DOUBLE);
  cubic_spline_transform(im_spline_coeff, im);

  /* Force iterator coordinates to be updated */
  UPDATE_ITERATOR_COORDS(imIter); 

  /* Resampling loop */
  while(imIter->index < imIter->size) {
    x = imIter->coordinates[0];
    y = imIter->coordinates[1]; 
    z = imIter->coordinates[2]; 
    _apply_affine_transform(&Tx, &Ty, &Tz, Tvox, x, y, z); 
    i1 = cubic_spline_sample3d(Tx, Ty, Tz, im_spline_coeff, mode_x, mode_y, mode_z); 
    if (cast_integer) {
      i1 = ROUND(i1);
      if (cast_integer == 2)
	if (i1 < 0)
	  i1 = 0;
     }

    /* Copy interpolated value into numpy array */
    py_i1 = PyFloat_FromDouble(i1); 
    PyArray_SETITEM(im_resampled, PyArray_ITER_DATA(imIter), py_i1); 
    Py_DECREF(py_i1); 
    
    /* Increment iterator */
    PyArray_ITER_NEXT(imIter); 

  }

  /* Free memory */
  Py_DECREF(imIter);
  Py_DECREF(im_spline_coeff); 
    
  return;
}

static inline void _apply_affine_transform(double* Tx, double* Ty, double* Tz, 
					   const double* Tvox,
					   size_t x, size_t y, size_t z)
{
  double* bufTvox = (double*)Tvox; 

  *Tx = (*bufTvox)*x; bufTvox++;
  *Tx += (*bufTvox)*y; bufTvox++;
  *Tx += (*bufTvox)*z; bufTvox++;
  *Tx += *bufTvox; bufTvox++;
  *Ty = (*bufTvox)*x; bufTvox++;
  *Ty += (*bufTvox)*y; bufTvox++;
  *Ty += (*bufTvox)*z; bufTvox++;
  *Ty += *bufTvox; bufTvox++;
  *Tz = (*bufTvox)*x; bufTvox++;
  *Tz += (*bufTvox)*y; bufTvox++;
  *Tz += (*bufTvox)*z; bufTvox++;
  *Tz += *bufTvox;

  return; 
}


/* 
   Convert an input grid coordinate x into another grid coordinate
   within [0, ddim], possibly using a reflection. This function
   implicitely assumes that -ddim < x < 2*ddim
 */
static inline int _mirrored_position(int x, unsigned int ddim)
{
  if (x < 0)
    return -x;
  else if (x > ddim) 
    return 2 * ddim - x;
  else
    return x;
}

/*
Depending on the chosen mode, mirror the position and set the weight.
*/
static inline int _apply_boundary_conditions(int mode, unsigned int ddim, 
					     double* x, double* w)
{
  int ok = 1; 
  unsigned int dim = ddim + 1; 
  int neg_ddim;
  unsigned int two_ddim;

  if (mode == 0) {
    if (*x < -1)
      ok = 0;
    else if (*x < 0) {
      *w = 1 + *x;
      *x = 0;
    }
    else if (*x > dim)
      ok = 0;
    else if (*x > ddim) {
      *w = dim - *x;
      *x = ddim;
    }
  }
  else if (mode == 1) {
    if (*x < 0)
      *x = 0;
    else if (*x > ddim)
      *x = ddim;
  }
  else{ /* mode==2 */
    neg_ddim = -ddim; 
    two_ddim = 2 * ddim; 
    if ((*x < neg_ddim) || (*x > two_ddim))
      ok = 0;
  }
  
  return ok; 
}

/* 
   Compute left and right cubic spline neighbors in the image grid
   mirrored once on each side. Returns 0 if no neighbor can be found.
*/
static inline int _mirror_grid_neighbors(double x, unsigned int ddim, 
					 int* nx, int* px)
{
  int ok = 0; 
  *px = (int)(x+ddim+2);
  if ((*px>=3) && (*px<=3*ddim)) {
    ok = 1; 
    *px = *px-ddim;
    *nx = *px-3;
  }		
  return ok; 
}



static inline int _neighbors_zero_outside(double x, unsigned int ddim, 
					  int* nx, int* px, 
					  double* weight)
{
  int ok = 0, aux;
  unsigned int dim = ddim+1;
  *weight = 1; 

  if ((x>-1) && (x<dim)) {
      ok = 1; 
      aux = (int)(x+2); 
      *nx = aux-3; 
      if (*nx<-1) { /* -1<x<0, nx==-2 */
	*weight = 1-x;
	*nx = -1;
      }
      *px = aux;
      if (*px>dim) { /* ddim<=x<dim, px=dim+1 */ 
	*weight = dim-x; 
	*px = dim; 
      }
    }

  return ok; 	
}


