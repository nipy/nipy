#include "fff_glm_twolevel.h"
#include "fff_base.h"
#include "fff_blas.h"

#include <stdlib.h>
#include <math.h>
#include <stdlib.h>

/*
  b, s2 are initialized using the values passed to the function.
  
  The function requires the projected pseudo-inverse matrix PpiX to be
  pre-calculated externally. It is defined by:

  PpiX = P * (X'X)^-1 X'
  where:

  P = Ip - A C' (C A C')^-1 C  with A = (X'X)^-1

  is the appropriate projector onto the constaint space, Cb=0. P is,
  in fact, orthogonal for the dot product defined by X'X. 

  PpiX is p x n. The equality PpiX*X=P is not checked. 
*/ 


fff_glm_twolevel_EM* fff_glm_twolevel_EM_new(size_t n, size_t p)
{
  fff_glm_twolevel_EM* thisone; 

  thisone = (fff_glm_twolevel_EM*)malloc(sizeof(fff_glm_twolevel_EM)); 

  if (thisone==NULL)
    return NULL; 

  thisone->n = n; 
  thisone->p = p;
  thisone->s2 = FFF_POSINF;
  
  thisone->b = fff_vector_new(p); 
  thisone->z = fff_vector_new(n); 
  thisone->vz = fff_vector_new(n); 
  thisone->Qz = fff_vector_new(n);

  return thisone;   
}

void fff_glm_twolevel_EM_delete(fff_glm_twolevel_EM* thisone)
{
  if (thisone==NULL)
    return; 
  fff_vector_delete(thisone->b); 
  fff_vector_delete(thisone->z);
  fff_vector_delete(thisone->vz); 
  fff_vector_delete(thisone->Qz);  
  free(thisone); 
}


void fff_glm_twolevel_EM_init(fff_glm_twolevel_EM* em)
{
  fff_vector_set_all(em->b, 0.0);
  em->s2 = FFF_POSINF;
  return; 
}


void fff_glm_twolevel_EM_run(fff_glm_twolevel_EM* em, const fff_vector* y, const fff_vector* vy, 
			     const fff_matrix* X, const fff_matrix* PpiX, unsigned int niter)
{
  unsigned int iter = 0;
  size_t n=X->size1, i;
  double *yi, *zi, *vyi, *vzi; 
  double w1, w2; 
  double m = 0.0; 


  while (iter < niter) {

    /*** E step ***/ 

    /* Compute current prediction estimate: z = X*b */ 
    fff_blas_dgemv(CblasNoTrans, 1.0, X, em->b, 0.0, em->z); 

    /* Posterior mean and variance of each "true" effect: 
         vz = 1/(1/vy + 1/s2) 
	 z =  vz * (y/vy + X*b/s2) */ 
    w2 = FFF_ENSURE_POSITIVE(em->s2);
    w2 = 1/w2; 
    for(i=0, yi=y->data, zi=em->z->data, vyi=vy->data, vzi=em->vz->data; 
	 i<n; 
	 i++, yi+=y->stride, zi+=em->z->stride, vyi+=vy->stride, vzi+=em->vz->stride) {
      w1 = FFF_ENSURE_POSITIVE(*vyi);
      w1 = 1/w1; 
      *vzi = 1/(w1+w2);
      *zi = *vzi * (w1*(*yi) + w2*(*zi)); 
    }

    /*** M step ***/ 

    /* Update effect: b = PpiX * z */ 
    fff_blas_dgemv(CblasNoTrans, 1.0, PpiX, em->z, 0.0, em->b); 
    
    /* Update variance: s2 = (1/n) [ sum((z-Xb).^2) + sum(vz) ] */ 
    fff_vector_memcpy(em->Qz, em->z);     
    fff_blas_dgemv(CblasNoTrans, 1.0, X, em->b, -1.0, em->Qz); /* Qz= Xb-z = Proj_X(z) - z */ 
    em->s2 = (fff_vector_ssd(em->Qz, &m, 1) + fff_vector_sum(em->vz)) / (long double)n;

    /*** Increment iteration number ***/
    iter ++; 
  }

  return;
}


/* 
   Log-likelihood computation. 

   ri = y - Xb
   -2 LL = n log(2pi) + \sum_i log (s^2 + si^2) + \sum_i ri^2/(s^2 + si^2)

   We omit the nlog(2pi) term as it is constant. 
*/ 
double fff_glm_twolevel_log_likelihood(const fff_vector* y, 
				       const fff_vector* vy, 
				       const fff_matrix* X, 
				       const fff_vector* b, 
				       double s2, 
				       fff_vector* tmp) 
{
  double LL = 0.0, w; 
  size_t n=X->size1, i;
  double *ri, *vyi; 

  /* Compute residuals: tmp = y - X b */ 
  fff_vector_memcpy(tmp, y);     
  fff_blas_dgemv(CblasNoTrans, -1.0, X, b, 1.0, tmp);   

  /* Incremental computation */ 
  for(i=0, ri=tmp->data, vyi=vy->data; i<n; i++, ri+=tmp->stride, vyi+=vy->stride) {
    w = *vyi + s2; 
    w = FFF_ENSURE_POSITIVE(w);
    LL += log(w); 
    LL += FFF_SQR(*ri)/w;

  }

  /* Finalize computation */ 
  LL *= -0.5;  

  return LL; 
}


