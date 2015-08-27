#include "joint_histogram.h"
#include "wichmann_prng.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>


#define SQR(a) ((a)*(a))
#define FLOOR(a)((a)>0.0 ? (int)(a):(((int)(a)-a)!= 0.0 ? (int)(a)-1 : (int)(a)))  
#define UROUND(a) ((int)(a+0.5))
#define ROUND(a)(FLOOR(a+0.5))

#ifdef _MSC_VER
#define inline __inline
#endif

static inline void _pv_interpolation(unsigned int i, 
				     double* H, unsigned int clampJ, 
				     const signed short* J, 
				     const double* W, 
				     int nn, 
				     void* params);
static inline void _tri_interpolation(unsigned int i, 
				      double* H, unsigned int clampJ, 
				      const signed short* J, 
				      const double* W, 
				      int nn, 
				      void* params);
static inline void _rand_interpolation(unsigned int i, 
				       double* H, unsigned int clampJ, 
				       const signed short* J, 
				       const double* W, 
				       int nn, 
				       void* params); 

/* 
   
JOINT HISTOGRAM COMPUTATION. 
  
iterI : assumed to iterate over a signed short encoded, possibly
non-contiguous array.

imJ_padded : assumed C-contiguous (last index varies faster) & signed
short encoded.

H : assumed C-contiguous. 

Tvox : assumed C-contiguous: 

  either a 3x4=12-sized array (or bigger) for an affine transformation

  or a 3xN array for a pre-computed transformation, with N equal to
  the size of the array corresponding to iterI (no checking done)

Negative intensities are ignored.  

*/

#define APPEND_NEIGHBOR(q, w)			\
  j = J[q];					\
  if (j>=0) {					\
    *bufJnn = j; bufJnn ++; 			\
    *bufW = w; bufW ++;				\
    nn ++; }


int joint_histogram(PyArrayObject* JH, 
		    unsigned int clampI, 
		    unsigned int clampJ,  
		    PyArrayIterObject* iterI,
		    const PyArrayObject* imJ_padded, 
		    const PyArrayObject* Tvox, 
		    long interp)
{
  const signed short* J=(signed short*)imJ_padded->data; 
  size_t dimJX=imJ_padded->dimensions[0]-2;
  size_t dimJY=imJ_padded->dimensions[1]-2; 
  size_t dimJZ=imJ_padded->dimensions[2]-2;  
  signed short Jnn[8]; 
  double W[8]; 
  signed short *bufI, *bufJnn; 
  double *bufW; 
  signed short i, j;
  size_t off;
  size_t u2 = imJ_padded->dimensions[2]; 
  size_t u3 = u2+1; 
  size_t u4 = imJ_padded->dimensions[1]*u2;
  size_t u5 = u4+1; 
  size_t u6 = u4+u2; 
  size_t u7 = u6+1; 
  double wx, wy, wz, wxwy, wxwz, wywz; 
  double W0, W2, W3, W4; 
  int nn, nx, ny, nz;
  double *H = (double*)PyArray_DATA(JH);  
  double Tx, Ty, Tz; 
  double *tvox = (double*)PyArray_DATA(Tvox); 
  void (*interpolate)(unsigned int, double*, unsigned int, const signed short*, const double*, int, void*); 
  void* interp_params = NULL; 
  prng_state rng; 


  /* 
     Check assumptions regarding input arrays. If it fails, the
     function will return -1 without doing anything else. 
     
     iterI : assumed to iterate over a signed short encoded, possibly
     non-contiguous array.
     
     imJ_padded : assumed C-contiguous (last index varies faster) & signed
     short encoded.
     
     H : assumed C-contiguous. 
     
     Tvox : assumed C-contiguous: 
     
       either a 3x4=12-sized array (or bigger) for an affine transformation

       or a 3xN array for a pre-computed transformation, with N equal
       to the size of the array corresponding to iterI (no checking
       done)
  
  */
  if (PyArray_TYPE(iterI->ao) != NPY_SHORT) {
    fprintf(stderr, "Invalid type for the array iterator\n");
    return -1; 
  }
  if ( (!PyArray_ISCONTIGUOUS(imJ_padded)) || 
       (!PyArray_ISCONTIGUOUS(JH)) ||
       (!PyArray_ISCONTIGUOUS(Tvox)) ) {
    fprintf(stderr, "Some non-contiguous arrays\n");
    return -1; 
  }

  /* Reset the source image iterator */
  PyArray_ITER_RESET(iterI);

  /* Set interpolation method */ 
  if (interp==0) 
    interpolate = &_pv_interpolation;
  else if (interp>0) 
    interpolate = &_tri_interpolation; 
  else { /* interp < 0 */ 
    interpolate = &_rand_interpolation;
    prng_seed(-interp, &rng); 
    interp_params = (void*)(&rng); 
  }

  /* Re-initialize joint histogram */ 
  memset((void*)H, 0, clampI*clampJ*sizeof(double));

  /* Looop over source voxels */
  while(iterI->index < iterI->size) {
  
    /* Source voxel intensity */
    bufI = (signed short*)PyArray_ITER_DATA(iterI); 
    i = bufI[0];

    /* Compute the transformed grid coordinates of current voxel */ 
    Tx = *tvox; tvox++;
    Ty = *tvox; tvox++;
    Tz = *tvox; tvox++; 

    /* Test whether the current voxel is below the intensity
       threshold, or the transformed point is completly outside
       the reference grid */
    if ((i>=0) && 
	(Tx>-1) && (Tx<dimJX) && 
	(Ty>-1) && (Ty<dimJY) && 
	(Tz>-1) && (Tz<dimJZ)) {
	
      /* 
	 Nearest neighbor (floor coordinates in the padded
	 image, hence +1). 
	 
	 Notice that using the floor function doubles excetution time.
	 
	 FIXME: see if we can replace this with assembler instructions. 
      */
      nx = FLOOR(Tx) + 1;
      ny = FLOOR(Ty) + 1;
      nz = FLOOR(Tz) + 1;
      
      /* The convention for neighbor indexing is as follows:
       *
       *   Floor slice        Ceil slice
       *
       *     2----6             3----7                     y          
       *     |    |             |    |                     ^ 
       *     |    |             |    |                     |
       *     0----4             1----5                     ---> x
       */
      
      /*** Trilinear interpolation weights.  
	   Note: wx = nnx + 1 - Tx, where nnx is the location in
	   the NON-PADDED grid */ 
      wx = nx - Tx; 
      wy = ny - Ty;
      wz = nz - Tz;
      wxwy = wx*wy;    
      wxwz = wx*wz;
      wywz = wy*wz;
      
      /*** Prepare buffers */ 
      bufJnn = Jnn;
      bufW = W; 
      
      /*** Initialize neighbor list */
      off = nx*u4 + ny*u2 + nz; 
      nn = 0; 
      
      /*** Neighbor 0: (0,0,0) */ 
      W0 = wxwy*wz; 
      APPEND_NEIGHBOR(off, W0); 
      
      /*** Neighbor 1: (0,0,1) */ 
      APPEND_NEIGHBOR(off+1, wxwy-W0);
      
      /*** Neighbor 2: (0,1,0) */ 
      W2 = wxwz-W0; 
      APPEND_NEIGHBOR(off+u2, W2);  
      
      /*** Neightbor 3: (0,1,1) */
      W3 = wx-wxwy-W2;  
      APPEND_NEIGHBOR(off+u3, W3);  
      
      /*** Neighbor 4: (1,0,0) */
      W4 = wywz-W0;  
      APPEND_NEIGHBOR(off+u4, W4); 
      
      /*** Neighbor 5: (1,0,1) */ 
      APPEND_NEIGHBOR(off+u5, wy-wxwy-W4);   
      
      /*** Neighbor 6: (1,1,0) */ 
      APPEND_NEIGHBOR(off+u6, wz-wxwz-W4);  
      
      /*** Neighbor 7: (1,1,1) */ 
      APPEND_NEIGHBOR(off+u7, 1-W3-wy-wz+wywz);  
      
      /* Update the joint histogram using the desired interpolation technique */ 
      interpolate(i, H, clampJ, Jnn, W, nn, interp_params); 
      
      
    } /* End of IF TRANSFORMS INSIDE */
    
    /* Update source index */ 
    PyArray_ITER_NEXT(iterI); 
    
  } /* End of loop over voxels */ 
  

  return 0; 
}


/* Partial Volume interpolation. See Maes et al, IEEE TMI, 2007. */ 
static inline void _pv_interpolation(unsigned int i, 
				     double* H, unsigned int clampJ, 
				     const signed short* J, 
				     const double* W, 
				     int nn, 
				     void* params) 
{ 
  int k;
  unsigned int clampJ_i = clampJ*i;
  const signed short *bufJ = J;
  const double *bufW = W; 

  for(k=0; k<nn; k++, bufJ++, bufW++) 
    H[*bufJ+clampJ_i] += *bufW;
 
  return; 
}

/* Trilinear interpolation. Basic version. */
static inline void _tri_interpolation(unsigned int i, 
				      double* H, unsigned int clampJ, 
				      const signed short* J, 
				      const double* W, 
				      int nn, 
				      void* params) 
{ 
  int k;
  unsigned int clampJ_i = clampJ*i;
  const signed short *bufJ = J;
  const double *bufW = W; 
  double jm, sumW; 
  
  for(k=0, sumW=0.0, jm=0.0; k<nn; k++, bufJ++, bufW++) {
    sumW += *bufW; 
    jm += (*bufW)*(*bufJ); 
  }
  if (sumW > 0.0) {
    jm /= sumW;
    H[UROUND(jm)+clampJ_i] += 1;
  }
  return; 
}

/* Random interpolation. */
static inline void _rand_interpolation(unsigned int i, 
				       double* H, unsigned int clampJ, 
				       const signed short* J, 
				       const double* W, 
				       int nn, 
				       void* params) 
{ 
  prng_state* rng = (prng_state*)params; 
  int k;
  unsigned int clampJ_i = clampJ*i;
  const double *bufW;
  double sumW, draw; 
  
  for(k=0, bufW=W, sumW=0.0; k<nn; k++, bufW++) 
    sumW += *bufW; 
  
  draw = sumW*prng_double(rng); 

  for(k=0, bufW=W, sumW=0.0; k<nn; k++, bufW++) {
    sumW += *bufW; 
    if (sumW > draw) 
      break; 
  }
    
  H[J[k]+clampJ_i] += 1;
  
  return; 
}


/* 
   A function to compute the weighted median in one-dimensional
   histogram.
 */
int L1_moments(double* n_, double* median_, double* dev_, 
	       const PyArrayObject* H)
{
  int i, med;
  double median, dev, n, cpdf, lim;
  const double *buf;
  const double* h;
  unsigned int size;
  unsigned int offset; 

  if (PyArray_TYPE(H) != NPY_DOUBLE) {
    fprintf(stderr, "Input array should be double\n");
    return -1; 
  }

  /* Initialize */
  h = (const double*)PyArray_DATA(H);
  size = PyArray_DIM(H, 0); 
  offset = PyArray_STRIDE(H, 0)/sizeof(double); 

  n = median = dev = 0; 
  cpdf = 0;
  buf = h;
  for (i=0; i<size; i++, buf+=offset) 
    n += *buf;
  
  /* Look for index i such that h(i-1) < n/2 and h(i) >= n/2 */
  if (n > 0) {
    
    lim = 0.5*n;
    i = 0;
    buf = h;
    cpdf = *buf;
    dev = 0;
    
    while (cpdf < lim) {
      i ++;
      buf += offset;
      cpdf += *buf;
      dev += - i*(*buf);
    }
    
    /* 
       We then have: i-1 < med < i and choose i as the median
       (alternatively, an interpolation between i-1 and i could be
       performed by linearly approximating the cumulative function).
       
       The L1 deviation reads:
       
       sum*E(|X-med|) = - sum_{i<=med} i h(i)            [1]
       
                      + sum_{i>med} i h(i)               [2]
     
                      + med * [2*cpdf(med) - sum]        [3]


       Term [1] is currently equal to `dev` variable. 
    */
    median = (double)i;
    dev += (2*cpdf - n)*median;
    med = i+1;
        
    /* Complete computation of the L1 deviation by computing the truncated mean [2]) */
    if (med < size) {
      buf = h + med*offset;
      for (i=med; i<size; i ++, buf += offset) 
	dev += i*(*buf);
    }
    
    dev /= n; 

  }

  n_[0] = n; 
  median_[0] = median; 
  dev_[0] = dev; 

  return 0;           
}



