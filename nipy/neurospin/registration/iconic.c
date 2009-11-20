#include "iconic.h"

#include <randomkit.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>


#define SQR(a) ((a)*(a))
#define FLOOR(a)((a)>0.0 ? (int)(a):(((int)(a)-a)!= 0.0 ? (int)(a)-1 : (int)(a)))  
#define UROUND(a) ((int)(a+0.5))
#define ROUND(a)(FLOOR(a+0.5))


/* 
   The following forces numpy to consider a PyArrayIterObject
   non-contiguous. Otherwise, coordinates won't be updated, apparently
   for computation time reasons.
*/
#define UPDATE_ITERATOR_COORDS(iter)		\
  iter->contiguous = 0;



static double _marginalize(double* h, 
			   const double* H, 
			   unsigned int clampI, 
			   unsigned int clampJ, 
			   int axis); 
static inline void _apply_affine_transform(double* Tx, 
					   double* Ty, 
					   double* Tz, 
					   const double* Tvox, 
					   size_t x, 
					   size_t y, 
					   size_t z); 
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



/* Numpy import */
void iconic_import_array(void) { 
  import_array(); 
  return;
}



/* 
   
SINGLE IMAGE HISTOGRAM COMPUTATION. 

This is not relevant to image registration but is useful for texture
analysis.
  
iter : assumed to iterate over a signed short encoded, possibly
non-contiguous array.

H : assumed C-contiguous. 

Negative intensities are ignored. 

*/

void histogram(double* H, 
	       unsigned int clamp, 
	       PyArrayIterObject* iter)
{
  signed short *buf;
  signed short i;

  /* Reset the source image iterator */
  PyArray_ITER_RESET(iter);

  /* Re-initialize joint histogram */ 
  memset((void*)H, 0, clamp*sizeof(double));

  /* Loop over source voxels */
  while(iter->index < iter->size) {
  
    /* Source voxel intensity */
    buf = (signed short*)PyArray_ITER_DATA(iter); 
    i = buf[0];

    /* Update the histogram only if the current voxel is below the
       intensity threshold */
    if (i>=0) 
      H[i]++; 
    
    /* Update source index */ 
    PyArray_ITER_NEXT(iter); 
    
  } /* End of loop over voxels */ 
  

  return; 
}

/*

  size should be odd numbers 

 */ 

void local_histogram(double* H, 
		     unsigned int clamp, 
		     PyArrayIterObject* iter, 
		     const unsigned int* size)
{
  PyArrayObject *block, *im = iter->ao; 
  PyArrayIterObject* block_iter; 
  unsigned int i, left, right, center, halfsize, dim, offset=0; 
  npy_intp block_dims[3];

  UPDATE_ITERATOR_COORDS(iter); 

  /* Compute block corners */ 
  for (i=0; i<3; i++) {
    center = iter->coordinates[i];
    halfsize = size[i]/2; 
    dim = PyArray_DIM(im, i);
  
    /* Left handside corner */ 
    if (center<halfsize)
      left = 0; 
    else
      left = center-halfsize; 

    /* Right handside corner (plus one)*/ 
    right = center+halfsize+1; 
    if (right>dim) 
      right = dim; 

    /* Block properties */ 
    offset += left*PyArray_STRIDE(im, i); 
    block_dims[i] = right-left;

  }

  /* Create the block as a vew and the block iterator */ 
  block = (PyArrayObject*)PyArray_New(&PyArray_Type, 3, block_dims, 
				      PyArray_TYPE(im), PyArray_STRIDES(im), 
				      (void*)(PyArray_DATA(im)+offset), 
				      PyArray_ITEMSIZE(im),
				      NPY_BEHAVED, NULL);
  block_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)block); 

  /* Compute block histogram */ 
  histogram(H, clamp, block_iter); 

  /* Free memory */ 
  Py_XDECREF(block_iter); 
  Py_XDECREF(block); 

  return; 
}		     
		     



/* 
   
JOINT HISTOGRAM COMPUTATION. 
  
iterI : assumed to iterate over a signed short encoded, possibly
non-contiguous array.

imJ_padded : assumed C-contiguous (last index varies faster) & signed
short encoded.

H : assumed C-contiguous. 

Tvox : assumed C-contiguous.

Negative intensities are ignored. 

*/

#define APPEND_NEIGHBOR(q, w)			\
  j = J[q];					\
  if (j>=0) {					\
    *bufJnn = j; bufJnn ++; 			\
    *bufW = w; bufW ++;				\
    nn ++; }


void joint_histogram(double* H, 
		     unsigned int clampI, 
		     unsigned int clampJ,  
		     PyArrayIterObject* iterI,
		     const PyArrayObject* imJ_padded, 
		     const double* Tvox, 
		     int interp)
{
  const signed short* J=(signed short*)imJ_padded->data; 
  size_t dimJX=imJ_padded->dimensions[0]-2, dimJY=imJ_padded->dimensions[1]-2, dimJZ=imJ_padded->dimensions[2]-2;  
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
  size_t x, y, z; 
  int nn, nx, ny, nz;
  double Tx, Ty, Tz; 
  void (*interpolate)(unsigned int, double*, unsigned int, const signed short*, const double*, int, void*); 
  void* interp_params = NULL; 
  rk_state rng; 

  /* Reset the source image iterator */
  PyArray_ITER_RESET(iterI);

  /* Set interpolation method */ 
  if (interp==0) 
    interpolate = &_pv_interpolation;
  else if (interp>0) 
    interpolate = &_tri_interpolation; 
  else { /* interp < 0 */ 
    interpolate = &_rand_interpolation;
    rk_seed(-interp, &rng); 
    interp_params = (void*)(&rng); 
  }

  /* Re-initialize joint histogram */ 
  memset((void*)H, 0, clampI*clampJ*sizeof(double));

  /* Looop over source voxels */
  while(iterI->index < iterI->size) {
  
    /* Source voxel intensity */
    bufI = (signed short*)PyArray_ITER_DATA(iterI); 
    i = bufI[0];

    /* Source voxel coordinates */
    x = iterI->coordinates[0];
    y = iterI->coordinates[1];
    z = iterI->coordinates[2];
    
    /* Compute the transformed grid coordinates of current voxel */ 
    _apply_affine_transform(&Tx, &Ty, &Tz, Tvox, x, y, z); 
    
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
  

  return; 
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
  rk_state* rng = (rk_state*)params; 
  int k;
  unsigned int clampJ_i = clampJ*i;
  const double *bufW;
  double sumW, draw; 
  
  for(k=0, bufW=W, sumW=0.0; k<nn; k++, bufW++) 
    sumW += *bufW; 
  
  draw = sumW*rk_double(rng); 

  for(k=0, bufW=W, sumW=0.0; k<nn; k++, bufW++) {
    sumW += *bufW; 
    if (sumW > draw) 
      break; 
  }
    
  H[J[k]+clampJ_i] += 1;
  
  return; 
}



static inline void _apply_affine_transform(double* Tx, double* Ty, double* Tz, 
					   const double* Tvox, size_t x, size_t y, size_t z)
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
  axis == 0 -> source histogram
  axis == 1 -> target histogram 
  
  returns the sum of H 

*/
static double _marginalize(double* h, const double* H, unsigned int clampI, unsigned int clampJ, int axis)
{
  int i, j; 
  const double *bufH = H; 
  double *bufh = h;
  double hij, sumH = 0.0; 
  
  if (axis == 0) {
    memset((void*)h, 0, clampI*sizeof(double));
    for (i=0; i<clampI; i++, bufh++) 
      for (j=0; j<clampJ; j++, bufH++) {
	hij = *bufH; 
	sumH += hij; 
	*bufh += hij; 
      }
  }
  else if (axis == 1) {
    memset((void*)h, 0, clampJ*sizeof(double));
    for (j=0; j<clampJ; j++, bufh++) {
      bufH = H + j; 
      for (i=0; i<clampI; i++, bufH+=clampJ) {
	hij = *bufH; 
	sumH += hij; 
	*bufh += hij; 
      }
    }
  }
  
  return sumH; 
}




/* =========================================================================
   HISTOGRAM-BASED SIMILARITY MEASURES 
   ========================================================================= */

#define TINY 1e-30
#define NICELOG(x)((x)>(TINY) ? log(x):log(TINY)) 


double correlation_coefficient(const double* H, unsigned int clampI, unsigned int clampJ, double* n)
{
  int i, j;
  double CC, mj, mi, mij, mj2, mi2, varj, vari, covij, na;
  double aux, auxj, auxi;
  const double *buf;

  /* Initialization */
  na = 0;
  mj  = 0;
  mi  = 0;
  mij = 0;
  mj2 = 0;
  mi2 = 0;
  
  /* Computes conditional moments */
  buf = H;
  for (i=0; i<clampI; i++) 
    for (j=0; j<clampJ; j++, buf++) {
      
      aux = *buf; /* aux == H(i,j) */
      auxj = j * aux;
      auxi = i * aux;
      
      na += aux;
      mj  += auxj;
      mi  += auxi;
      mj2 += j * auxj;
      mi2 += i * auxi;
      mij += i * auxj;
      
  }
  
  /* Test if the dataset is empty */
  if (na <= 0) {
    *n = 0; 
    return 0;
  }
  
  /* Normalization */
  mj  = mj / na;
  mi  = mi / na;
  mj2 = mj2 / na;
  mi2 = mi2 / na;
  mij = mij / na;
  
  /* Compute covariance and variance */
  covij = mij - mj * mi;
  covij = SQR(covij);
  varj  = mj2 - mj * mj;
  vari  = mi2 - mi * mi;
  aux = varj * vari;
  
  /* Check the variance product is non zero */
  if (aux <= 0) {
    *n = na;
    return 0;
  }
  
  /* Return the correlation coefficient */
  CC = covij / aux;
  
  *n = na; 
  return CC;
     
}


double correlation_ratio(const double* H, unsigned int clampI, unsigned int clampJ, double* n)
{
  int j;
  double CR, na, mean, var, cvar, nJ;
  double moments[5]; 

  /* Initialization */
  na = mean = var = cvar = 0;

  /* Compute conditional moments */
  for (j=0; j<clampJ; j++) {
    L2_moments_with_stride (H+j, clampI, clampJ, moments);
    nJ = moments[0]; 
    na += nJ;
    mean += moments[3]; 
    var += moments[4]; 
    cvar += nJ*moments[2];
  }
    
  /* Compute total moments */
  if (na <= 0) {
    *n = 0;
    return 0;
  }
  mean /= na;
  var = var/na - mean*mean;
  cvar /= na;

  /* Test on total variance */
  if (var<=0) {
    *n = na; 
    return 0;
  }
  
  /* Correlation ratio */
  *n = na; 
  CR = 1 - cvar/var;

  return CR;
  
}



double correlation_ratio_L1(const double* H, double* hI, unsigned int clampI, unsigned int clampJ, double* n)
{
  int j;
  double na, med, dev, cdev, nJ, mJ, dJ;
  double moments[3]; 

  /* Initialization */
  na = cdev = 0;

  /* Compute conditional moments */  
  for (j=0; j<clampJ; j++) {
    L1_moments_with_stride (H+j, clampI, clampJ, moments); 
    nJ = moments[0]; 
    mJ = moments[1]; 
    dJ = moments[2]; 
    cdev += nJ*dJ;
    na += nJ;
  }
  
  /* Conditional dispersion */
  if (na <= 0.0) {
    *n = 0; 
    return 0.0;
  }
  cdev /= na;

  /* Total dispersion */
  _marginalize(hI, H, clampI, clampJ, 0); 
  L1_moments_with_stride (hI, clampI, 1, moments); 
  med = moments[1]; 
  dev = moments[2]; 

  /* Test on total dispersion */
  *n = na; 
  if (dev==0.0)
    return 0.0;
  else
    return(1 - SQR(cdev)/SQR(dev)); /* Squaring for comparison with CR(L2) */  
}



double joint_entropy(const double* H, unsigned int clampI, unsigned int clampJ)
{
  double n; 
  double entIJ = entropy(H, clampI*clampJ, &n); 
  return entIJ; 
}


double conditional_entropy(const double* H, double* hJ, unsigned int clampI, unsigned int clampJ)
{
  double n; 
  double entIJ, entJ;  
  _marginalize(hJ, H, clampI, clampJ, 1); 
  entIJ = entropy(H, clampI*clampJ, &n); 
  entJ = entropy(hJ, clampJ, &n); 
  return(entIJ - entJ); /* Entropy of I given J */ 
}


double mutual_information(const double* H, double* hI, unsigned int clampI, double* hJ, unsigned int clampJ, double* n)
{
  double entIJ, entI, entJ; 
  _marginalize(hI, H, clampI, clampJ, 0); 
  _marginalize(hJ, H, clampI, clampJ, 1); 
  entI = entropy(hI, clampI, n); 
  entJ = entropy(hJ, clampJ, n); 
  entIJ = entropy(H, clampI*clampJ, n); 
  return(entI + entJ - entIJ); 
}

/*
  Normalized mutual information as advocated by Studholme, 98. 

  NMI = 2*(1 - H(I,J)/[H(I)+H(J)])
  
*/

double normalized_mutual_information(const double* H, double* hI, unsigned int clampI, double* hJ, unsigned int clampJ, double *n)
{
  double entIJ, entI, entJ, aux; 
  _marginalize(hI, H, clampI, clampJ, 0); 
  _marginalize(hJ, H, clampI, clampJ, 1); 
  entI = entropy(hI, clampI, n); 
  entJ = entropy(hJ, clampJ, n); 
  entIJ = entropy(H, clampI*clampJ, n);
  aux = entI + entJ; 
  if (aux > 0.0) 
    return(2*(1-entIJ/aux));
  else 
    return 0.0; 
}


/*

Supervised mutual information. 
See Roche, PhD dissertation, University of Nice-Sophia Antipolis, 2001. 

*/ 

double supervised_mutual_information(const double* H, const double* F, 
				     double* fI, unsigned int clampI, double* fJ, unsigned int clampJ, 
				     double* n)
{
  const double *bufH = H, *bufF = F; 
  double *buf_fI, *buf_fJ; 
  int i, j; 
  double hij, fij, fi, fj, aux, sumF, na = 0.0, SMI = 0.0; 

  sumF = _marginalize(fI, F, clampI, clampJ, 0); 
  sumF = _marginalize(fJ, F, clampI, clampJ, 1); 

  for (i=0, buf_fI=fI; i<clampI; i++, buf_fI++) {
    fi = *buf_fI / sumF; /* HACK to implicitely normalize F */ 
    for (j=0, buf_fJ=fJ; j<clampJ; j++, buf_fJ++, bufH++, bufF++) {
      hij = *bufH;
      na += hij;  
      fj = *buf_fJ;
      fij = *bufF;

      /* If fi=0 or fj=0, then fij = 0. We conventionally set
	 fij/(fi*fj) to zero but this might be problematic */ 
      aux = fi * fj; 
      if (aux > 0) 
	aux = fij / aux; 
      else 
	aux = 0.0;  
      SMI += hij * NICELOG(aux); 
    }
  }

  *n = na;

  if (*n>0.0) 
    SMI /= *n; 

  return SMI; 
}



void drange(const double* h, unsigned int size, double* res)
{
  unsigned int i, left;
  double *buf;

  /* Find the first value from the left with non-zero mass */
  buf = (double*)h;
  for (i=0; i<size; i++, buf++) {
    if (*buf > 0.0) 
      break; 
  }
  left = i; 
  res[0] = (double)i;
  
  /* Find the first value from the right with non-zero mass */
  buf = (double*)h; 
  buf += size-1; 
  for (i=(size-1); i>=left; i--, buf--) {
    if (*buf > 0.0) 
      break; 
  }
  res[1] = (double)i;
 
  return; 
}

/* 
   First loop: compute the histogram sum 
   Second loop: compute actual entropy 
*/ 
double entropy(const double* h, unsigned int size, double* n)
{
  double E=0.0, sumh=0.0, aux; 
  unsigned int i; 
  double *buf;

  buf = (double*)h; 
  for (i=0; i<size; i++, buf++) 
    sumh += *buf; 

  if (sumh <= 0) {
    *n = 0; 
    return 0.0; 
  }

  buf = (double*)h; 
  for (i=0; i<size; i++, buf++) {
    aux = *buf / sumh; 
    E -= aux * NICELOG(aux); 
  }

  *n = sumh; 
  return E; 
}


/* res must pre-allocated with size=5 */ 
void L2_moments_with_stride (const double * h, unsigned int size, unsigned int stride, 
			     double* res)
{
  unsigned int i;
  double mean, var, n, aux, aux2; 
  const double *buf;
  double *bufres; 

  n = mean = var = 0;
  buf = h;
  for (i=0; i<size; i++, buf+=stride) {
    aux = *buf; 
    n += aux;
    aux2 = i*aux;
    mean += aux2;
    var += i*aux2;
  }

  bufres = res + 3; *bufres = mean; 
  bufres ++; *bufres = var; 
  
  if (n>0) {
    mean /= n;
    var = var/n - mean*mean;
  }
  
  bufres = res; *bufres = n; 
  bufres ++; *bufres = mean; 
  bufres ++; *bufres = var; 

  return;
}


/* res must be preallocated with size=3 */ 
void L1_moments_with_stride (const double * h, unsigned int size, unsigned int stride, 
			     double* res)
{
  int i, med;
  double median, dev, n, cpdf, lim;
  const double *buf;
  double* bufres; 

  /* Initialisation au cas ou */
  n = median = dev = 0; 
  cpdf = 0;
  buf = h;
  for (i=0; i<size; i++, buf+=stride) {
    n += *buf;
  }
  
  if (n > 0) {
  
    /* On cherche la valeur i telle que h(i-1) < n/2 
       et h(i) >= n/2 */
    lim = 0.5*n;
    
    i = 0;
    buf = h;
    cpdf = *buf;
    dev = 0;
    
    while (cpdf < lim) {
      i ++;
      buf += stride;
      cpdf += *buf;
      dev += - i*(*buf);
    }
    
    /* 
       On a alors i-1 < med < i. On prend i comme valeur mediane, 
       meme s'il serait possible de choisir une valeur intermediaire
       entre i-1 et i, en approchant la fonction de repartition 
       par une droite. 
       
       La dispersion L1 est donnee par :
       
       sum*E(|X-med|) = - sum_{i<=med} i h(i)              [1]
       
                      + sum_{i>med} i h(i)               [2]
     
                      + med * [2*cpdf(med) - sum]        [3]


       Le terme [1] est actuellement egal a la variable dev.
       Le terme cpdf(med) est actuellement egal a la var. cpdf.
    */
    
    median = (double)i;
    dev += (2*cpdf - n)*median;
    med = i+1;
    
    
    /* Pour achever le calcul de la deviation L1, il suffit de calculer 
       la moyenne tronquee (entre i et la dimension) (cf. [2]) */
    
    if (med < size) {
      buf = h + med*stride;
      for (i=med; i<size; i ++, buf += stride) 
	dev += i*(*buf);
    }
    
    dev /= n; 

  }

  bufres = res; *bufres = n; 
  bufres ++; *bufres = median; 
  bufres ++; *bufres = dev; 

  return;           
}



