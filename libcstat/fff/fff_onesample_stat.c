#include "fff_onesample_stat.h"
#include "fff_base.h"
#include "fff_blas.h"

#include <stdlib.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>

#define EL_LDA_TOL 1e-5
#define EL_LDA_ITERMAX 100
#define MIN_RELATIVE_VAR_FFX 1e-4

/* Dummy structure for sorting */ 
typedef struct{
  double x; 
  size_t i; 
} fff_indexed_data;

/* Static structure for empirical MFX stats */ 
typedef struct{
  fff_vector* w; /* weights */ 
  fff_vector* z; /* centers */ 
  fff_matrix* Q; 
  fff_vector* tvar; /* low thresholded variances */  
  fff_vector* tmp1; 
  fff_vector* tmp2; 
  fff_indexed_data* idx; 
  unsigned int* niter; 
} fff_onesample_mfx;


/* Declaration of static functions */ 

/** Pure RFX analysis **/
static double _fff_onesample_mean(void* params, const fff_vector* x, double base); 
static double _fff_onesample_median(void* params, const fff_vector* x, double base); 
static double _fff_onesample_student(void* params, const fff_vector* x, double base); 
static double _fff_onesample_laplace(void* params, const fff_vector* x, double base); 
static double _fff_onesample_tukey(void* params, const fff_vector* x, double base); 
static double _fff_onesample_sign_stat(void* params, const fff_vector* x, double base); 
static double _fff_onesample_wilcoxon(void* params, const fff_vector* x, double base); 
static double _fff_onesample_elr(void* params, const fff_vector* x, double base); 
static double _fff_onesample_grubb(void* params, const fff_vector* x, double base); 
static void _fff_absolute_residuals(fff_vector* r, const fff_vector* x, double base);
static double _fff_el_solve_lda(fff_vector* c, const fff_vector* w); 

/** Normal MFX analysis **/
static double _fff_onesample_LR_gmfx(void* params, const fff_vector* x, const fff_vector* var, double base); 
static double _fff_onesample_mean_gmfx(void* params, const fff_vector* x, const fff_vector* var, double base); 
static void _fff_onesample_gmfx_EM(double* m, double* v, 
				   const fff_vector* x, const fff_vector* var, 
				   unsigned int niter, int constraint);
static double _fff_onesample_gmfx_nll(const fff_vector* x, const fff_vector* var, double m, double v);

/** Empirical MFX analysis **/
static fff_onesample_mfx* _fff_onesample_mfx_new(unsigned int n, unsigned int* niter, int flagIdx); 
static void _fff_onesample_mfx_delete(fff_onesample_mfx* thisone); 
static double _fff_onesample_mean_mfx(void* params, const fff_vector* x, const fff_vector* var, double base); 
static double _fff_onesample_median_mfx(void* params, const fff_vector* x, const fff_vector* var, double base); 
static double _fff_onesample_sign_stat_mfx(void* params, const fff_vector* x, const fff_vector* var, double base); 
static double _fff_onesample_wilcoxon_mfx(void* params, const fff_vector* x, const fff_vector* var, double base); 
static double _fff_onesample_LR_mfx(void* params, const fff_vector* x, const fff_vector* var, double base); 
static void _fff_onesample_mfx_EM(fff_onesample_mfx* Params, 
				   const fff_vector* x, const fff_vector* var, 
				   int constraint); 
static void _fff_onesample_mfx_EM_init(fff_onesample_mfx* Params, 
					const fff_vector* x, int flag);
static double _fff_onesample_mfx_nll(fff_onesample_mfx* Params, const fff_vector* x);


/** Low level for qsort **/ 
static int _fff_abs_comp(const void * x, const void * y); 
static int _fff_indexed_data_comp(const void * x, const void * y);
static void _fff_sort_z(fff_indexed_data* idx, fff_vector* tmp1, fff_vector* tmp2, 
			 const fff_vector* z, const fff_vector* w); 


fff_onesample_stat* fff_onesample_stat_new(unsigned int n, fff_onesample_stat_flag flag, double base)
{
  fff_onesample_stat* thisone = (fff_onesample_stat*)malloc(sizeof(fff_onesample_stat)); 

  if (thisone == NULL) 
    return NULL; 

  /* Fields */ 
  thisone->flag = flag; 
  thisone->base = base; 
  thisone->params = NULL; 
  
  /* Switch (possibly overwrite the 'par' field)*/ 
   switch (flag) {
     
   case FFF_ONESAMPLE_EMPIRICAL_MEAN:
     thisone->compute_stat = &_fff_onesample_mean;
     break;

   case FFF_ONESAMPLE_EMPIRICAL_MEDIAN:
     thisone->params = (void*) fff_vector_new(n); 
     thisone->compute_stat = &_fff_onesample_median;
     break;

   case FFF_ONESAMPLE_STUDENT:
     thisone->compute_stat = &_fff_onesample_student;
     break;

   case FFF_ONESAMPLE_LAPLACE:
     thisone->params = (void*) fff_vector_new(n); 
     thisone->compute_stat = &_fff_onesample_laplace;
     break;

   case FFF_ONESAMPLE_TUKEY:
     thisone->params = (void*) fff_vector_new(n); 
     thisone->compute_stat = &_fff_onesample_tukey;
     break;

   case FFF_ONESAMPLE_SIGN_STAT:
     thisone->compute_stat = &_fff_onesample_sign_stat;
     break;

   case FFF_ONESAMPLE_WILCOXON:
     thisone->params = (void*) fff_vector_new(n); 
     thisone->compute_stat = &_fff_onesample_wilcoxon;
     break;

   case FFF_ONESAMPLE_ELR:
     thisone->params = (void*) fff_vector_new(n); 
     thisone->compute_stat = &_fff_onesample_elr;
     break;

   case FFF_ONESAMPLE_GRUBB:
     thisone->compute_stat = &_fff_onesample_grubb;
     break;

   default:
     FFF_ERROR("Unrecognized statistic", EINVAL);
     break; 

   } /* End switch */ 
   
   return thisone; 
}



void fff_onesample_stat_delete(fff_onesample_stat* thisone)
{
  if (thisone == NULL) 
    return; 

  /* Switch */ 
  switch (thisone->flag) {

  default:
    break;

  case FFF_ONESAMPLE_LAPLACE:
  case FFF_ONESAMPLE_TUKEY:
  case FFF_ONESAMPLE_WILCOXON:
  case FFF_ONESAMPLE_ELR:
    fff_vector_delete((fff_vector*)thisone->params); 
    break; 

  } /* End switch */ 
  
  free(thisone); 
}


double fff_onesample_stat_eval(fff_onesample_stat* thisone, const fff_vector* x)
{
  double t; 
  t = thisone->compute_stat(thisone->params, x, thisone->base); 
  return t; 
}



/********************************** SAMPLE MEAN *******************************/ 
static double _fff_onesample_mean(void* params, const fff_vector* x, double base)
{
  double aux; 
  if (params != NULL) 
    return FFF_NAN; 
  aux = fff_vector_sum(x)/(long double)x->size - base;
  return aux; 
}



/********************************** SAMPLE MEDIAN ****************************/ 
static double _fff_onesample_median(void* params, const fff_vector* x, double base)
{
  double aux; 
  fff_vector* tmp = (fff_vector*)params; 

  fff_vector_memcpy(tmp, x); 
  aux = fff_vector_median(tmp) - base;
  return aux;   
}


/********************************** STUDENT STATISTIC ****************************/ 

static double _fff_onesample_student(void* params, const fff_vector* x, double base)
{
  double m, std, aux;
  int sign; 
  size_t n = x->size;

  if (params != NULL) 
    return FFF_NAN; 
  std = sqrt(fff_vector_ssd(x, &m, 0)/(long double)x->size);  
  aux = sqrt((double)(n-1))*(m-base);
  sign = (int) FFF_SIGN(aux);
  if (sign == 0) /* Sample mean equals baseline, return zero */ 
    return 0.0; 

  aux = aux / std; 
  if (sign > 0)
    if (aux < FFF_POSINF)
      return aux;
    else 
      return FFF_POSINF;  
  else 
    if (aux > FFF_NEGINF)
      return aux; 
    else 
      return FFF_NEGINF; 
}


/********************************** LAPLACE STATISTIC ****************************/ 

static double _fff_onesample_laplace(void* params, const fff_vector* x, double base)
{
  double s, s0, aux;
  int sign; 
  size_t n = x->size; 
  fff_vector* tmp = (fff_vector*)params; 

  fff_vector_memcpy(tmp, x); 
  aux = fff_vector_median(tmp); 
  s = fff_vector_sad(x, aux)/(long double)x->size; 
  s0 = fff_vector_sad(x, base)/(long double)x->size; 
  s0 = FFF_MAX(s0, s); /* Ensure s0 >= s */ 
  
  aux -= base; 
  sign = FFF_SIGN(aux); 
  if (sign == 0) /* Sample median equals baseline, return zero */ 
    return 0.0; 

  aux = sqrt(2*n*log(s0/s)); 
  if (aux < FFF_POSINF)
    return (sign * aux); 
  else if (sign > 0)
    return FFF_POSINF; 
  else 
    return FFF_NEGINF; 
}


/********************************** TUKEY STATISTIC ******************************/ 

static void _fff_absolute_residuals(fff_vector* r, const fff_vector* x, double base)
{
  size_t i, n = x->size;
  double aux; 
  double *bufX = x->data, *bufR = r->data; 
  
  for(i=0; i<n; i++, bufX+=x->stride, bufR+=r->stride) {
    aux = *bufX - base; 
    *bufR = FFF_ABS(aux); 
  }

  return; 
}

static double _fff_onesample_tukey(void* params, const fff_vector* x, double base)
{
  double s, s0, aux;
  int sign; 
  size_t n = x->size; 
  fff_vector* tmp = (fff_vector*)params;

  fff_vector_memcpy(tmp, x); 
  aux = fff_vector_median(tmp); 
 
  /* Take the median of absolute residuals |x_i-median| */
  _fff_absolute_residuals(tmp, x, aux); 
  s = fff_vector_median(tmp); 

  /* Take the median of absolute residuals |x_i-base| */
  _fff_absolute_residuals(tmp, x, base); 
  s0 = fff_vector_median(tmp); 
  s0 = FFF_MAX(s0, s); /* Ensure s0 >= s */ 

  aux -= base; /* aux == median(x) - base */
  sign = FFF_SIGN(aux); 
  if (sign == 0) /* Sample median equals baseline, return zero */ 
    return 0.0; 
  
  aux = sqrt(2*n*log(s0/s)); 
  if (aux < FFF_POSINF)
    return (sign * aux); 
  else if (sign > 0)
    return FFF_POSINF; 
  else 
    return FFF_NEGINF; 
}


/********************************** SIGN STATISTIC ****************************/ 

static double _fff_onesample_sign_stat(void* params, const fff_vector* x, double base)
{
  size_t i, n = x->size;
  double rp = 0.0, rm = 0.0, aux;
  double* buf = x->data; 
 
  if (params != NULL) 
    return FFF_NAN; 
  for (i=0; i<n; i++, buf+=x->stride) {
    aux = *buf - base; 
    if (aux > 0.0)
      rp ++;
    else if (aux < 0.0) 
      rm ++; 
    else { /* in case the sample value is exactly zero */ 
      rp += .5; 
      rm += .5;
    }
  }
  
  return (rp-rm)/(double)n; 
}



/********************* WILCOXON (SIGNED RANK) STATISTIC *********************/ 

static int _fff_abs_comp(const void * x, const void * y)
{
  int ans = 1;
  double xx = *((double*)x);
  double yy = *((double*)y);

  xx = FFF_ABS(xx);
  yy = FFF_ABS(yy); 

  if (yy > xx) {
    ans = -1;
    return ans;
  }
  if (yy == xx)
    ans = 0;

  return ans;
}

static double _fff_onesample_wilcoxon(void* params, const fff_vector* x, double base)
{
  size_t i, n = x->size; 
  double t = 0.0;
  double* buf; 
  fff_vector* tmp = (fff_vector*)params;
 
  /* Compute the residuals wrt baseline */ 
  fff_vector_memcpy(tmp, x); 
  fff_vector_add_constant(tmp, -base);

  /* Sort the residuals in terms of their ABSOLUTE values 
     NOTE: tmp needs be contiguous -- and it is, if allocated using fff_onesample_stat_new */ 
  qsort (tmp->data, n, sizeof(double), &_fff_abs_comp);

  /* Compute the sum of ranks multiplied by corresponding elements' signs */ 
  buf = tmp->data; 
  for(i=1; i<=n; i++, buf++) /* Again buf++ works IFF tmp is contiguous */ 
    t += (double)i * FFF_SIGN(*buf); 

  /* Normalization to have the stat range in [-1,1] */ 
  /*  t /= (double)((n*(n+1))/2);*/

  /* Normalization */ 
  t /= ((double)(n*n)); 

  return t; 
}

/************************ EMPIRICAL LIKELIHOOD STATISTIC **********************/ 

static double _fff_onesample_elr(void* params, const fff_vector* x, double base)
{
  size_t i, n = x->size; 
  double lda, aux, nwi;
  int sign; 
  fff_vector* tmp = (fff_vector*)params; 
  double* buf; 

  /* Compute: tmp = x-base */
  fff_vector_memcpy(tmp, x); 
  fff_vector_add_constant(tmp, -base); 
  aux = fff_vector_sum(tmp)/(long double)tmp->size;
  sign = FFF_SIGN(aux);

  /* If sample mean equals baseline, return zero */ 
  if (sign == 0) 
    return 0.0; 
  
  /* Find the Lagrange multiplier corresponding to the constrained
     empirical likelihood maximization problem */ 
  lda = _fff_el_solve_lda(tmp, NULL);
  if (lda >= FFF_POSINF) {
    if (sign > 0) 
      return FFF_POSINF; 
    else 
      return FFF_NEGINF; 
  }
  
  /* Compute the log empirical likelihood ratio, log lda = \sum_i \log(nw_i) */ 
  buf = x->data; 
  aux = 0.0;
  for(i=0; i<n; i++, buf+=x->stride) {
    nwi = 1/(1 + lda*(*buf-base)); 
    nwi = FFF_MAX(nwi, 0.0); 
    aux += log(nwi); 
  }

  /* We output \sqrt{-2\log\lambda} multiplied by the effect's sign */ 
  aux = -2.0 * aux; 
  aux = sqrt(FFF_MAX(aux, 0.0)); 

  if (aux < FFF_POSINF)
    return (sign*aux); 
  else if (sign > 0)
    return FFF_POSINF; 
  else 
    return FFF_NEGINF; 
}

/*
  Solve the equation: 
  
  sum(wi*ci/(lda*ci+1)) = 0
  
  where the unknown is lda and ci is the constraint, e.g. ci = xi-m.
  In standard RFX context, wi is uniformly constant, while in MFX
  context it may vary from one datapoint to another.
  
  By transforming ci into -1./ci, the equation becomes: 
  
  sum(wi/ (lda-ci)) = 0 

*/ 
static double _fff_el_solve_lda(fff_vector* c, const fff_vector* w)
{
  size_t i, n = c->size;
  unsigned int iter = 0; 
  double aux, g, dg, lda, lda0 = FFF_NEGINF, lda1 = FFF_POSINF, ldac, err;
  double *buf, *bufW; 

  /* Transform the constraint vector: c = -1./c and find the max and
     min elements of c such that c(i)<0 and c(i)>0, respectively */
  buf = c->data; 
  for (i=0; i<n; i++, buf+=c->stride) {
    aux = *buf; 
    aux = -1.0/aux; 
    *buf = aux; /* Vector values are overwritten */
    if ((aux<0.0) && (aux>lda0))
      lda0 = aux; 
    else if ((aux>0.0) && (aux<lda1))
      lda1 = aux; 
  }
  
  /* Return infinity if either lda0 or lda1 are not finite */ 
  if (!(lda0>FFF_NEGINF) ||  !(lda1<FFF_POSINF))
    return FFF_POSINF; 
  
  /* Initial guess for lda */ 
  lda = .5*(lda0+lda1);
  err = lda1 - lda0; 
  
  /* Solve the equation: 
     0 = g(lda) = sum(1./(lda-c(i))
     in the range lda0 < lda < lda1 
  We use a Newton algorithm */ 
  while(err > EL_LDA_TOL) {

    iter ++; 
    if (iter > EL_LDA_ITERMAX)
      break;
    
    /* Compute: 
       g(lda) = \sum_i w_i / (lda - c_i) 
       dg(lda) = -\sum_i w_i / (lda - c_i)^2  */ 
    g = 0.0; 
    dg = 0.0; 
    buf = c->data; 
    if (w == NULL) {
      for (i=0; i<n; i++, buf+=c->stride) { 
	aux = 1/(lda-*buf);
	g += aux; 
	dg += FFF_SQR(aux); 
      }
    }
    else {
      bufW = w->data; 
      for (i=0; i<n; i++, buf+=c->stride, bufW+=w->stride) { 
	aux = 1/(lda-*buf);
	g += *bufW * aux; 
	dg += *bufW * FFF_SQR(aux); 
      }
    }

    /* Update brakets */ 
    if (g > 0.0) 
      lda0 = lda; 
    else if (g < 0.0) 
      lda1 = lda; 

    /* Accept the Newton update if it falls within the brakets */ 
    ldac = lda + (g/dg);
    if ((lda0 < lda) && (lda < lda1)) 
      lda = ldac; 
    else 
      lda = .5*(lda0+lda1); 
    
    /* Error update */ 
    err = lda1 - lda0; 

  }

  return lda; 
}


/******************************* GRUBB STATISTIC *******************************/ 

static double _fff_onesample_grubb(void* params, const fff_vector* x, double base)
{
  size_t i;
  double t=0.0, mean, std, inv_std, ti;
  double *buf = x->data; 

  if (params != NULL)
    return FFF_NAN; 
  base = 0; 

  /* Compute the mean and std deviation */ 
  std = sqrt(fff_vector_ssd(x, &mean, 0)/(long double)x->size); 
  inv_std = 1/std; 
  if (t >= FFF_POSINF)
    return 0.0; 

  /* Compute the max of Studentized datapoints */
  for (i=0; i<x->size; i++, buf+=x->stride) {
    ti = (*buf-mean) * inv_std;
    ti = FFF_ABS(ti);
    if (ti > t)
      t = ti; 
  }

  return t; 
}





/*****************************************************************************************/
/*                       Mixed-effect statistic structure                                */
/*****************************************************************************************/

fff_onesample_stat_mfx* fff_onesample_stat_mfx_new(unsigned int n, fff_onesample_stat_flag flag, double base)
{
  fff_onesample_stat_mfx* thisone = (fff_onesample_stat_mfx*)malloc(sizeof(fff_onesample_stat_mfx)); 

  if (thisone == NULL) 
    return NULL; 

  /* Fields */ 
  thisone->flag = flag; 
  thisone->base = base; 
  thisone->empirical = 1; 
  thisone->niter = 0; 
  thisone->constraint = 0; 
  thisone->params = NULL; 
  
  /* Switch (possibly overwrite the 'par' field)*/ 
   switch (flag) {
     
   case FFF_ONESAMPLE_STUDENT_MFX:
     thisone->empirical = 0; 
     thisone->compute_stat = &_fff_onesample_LR_gmfx;
     thisone->params = (void*)(&(thisone->niter)); 
     break;

   case FFF_ONESAMPLE_GAUSSIAN_MEAN_MFX:
     thisone->empirical = 0; 
     thisone->compute_stat = &_fff_onesample_mean_gmfx;
     thisone->params = (void*)(&(thisone->niter)); 
     break;

   case FFF_ONESAMPLE_EMPIRICAL_MEAN_MFX:
     thisone->compute_stat = &_fff_onesample_mean_mfx;
     thisone->params = (void*)_fff_onesample_mfx_new(n, &(thisone->niter), 0);
     break;

   case FFF_ONESAMPLE_EMPIRICAL_MEDIAN_MFX:
     thisone->compute_stat = &_fff_onesample_median_mfx;
     thisone->params = (void*)_fff_onesample_mfx_new(n, &(thisone->niter), 1);
     break;

   case FFF_ONESAMPLE_SIGN_STAT_MFX:
     thisone->compute_stat = &_fff_onesample_sign_stat_mfx;
     thisone->params = (void*)_fff_onesample_mfx_new(n, &(thisone->niter), 0);
     break;

   case FFF_ONESAMPLE_WILCOXON_MFX:
     thisone->compute_stat = &_fff_onesample_wilcoxon_mfx;
     thisone->params = (void*)_fff_onesample_mfx_new(n, &(thisone->niter), 1);
     break;

   case FFF_ONESAMPLE_ELR_MFX:
     thisone->compute_stat = &_fff_onesample_LR_mfx;
     thisone->params = (void*)_fff_onesample_mfx_new(n, &(thisone->niter), 0);
     break;
     
   default:
     FFF_ERROR("Unrecognized statistic", EINVAL);
     break; 
  
   } /* End switch */ 
   
   return thisone; 
}

void fff_onesample_stat_mfx_delete(fff_onesample_stat_mfx* thisone)
{
  if (thisone == NULL) 
    return; 
  
  if (thisone->empirical) 
    _fff_onesample_mfx_delete((fff_onesample_mfx*)thisone->params); 
  
  free(thisone); 
  return; 
}


static fff_onesample_mfx* _fff_onesample_mfx_new(unsigned int n, unsigned int* niter, int flagIdx) 
{
  fff_onesample_mfx* thisone;

  thisone = (fff_onesample_mfx*)malloc(sizeof(fff_onesample_mfx));
  thisone->w = fff_vector_new(n); 
  thisone->z = fff_vector_new(n); 
  thisone->Q = fff_matrix_new(n, n); 
  thisone->tvar = fff_vector_new(n);
  thisone->tmp1 = fff_vector_new(n); 
  thisone->tmp2 = fff_vector_new(n);
  thisone->idx = NULL;
  thisone->niter = niter; 
  
  if (flagIdx == 1) 
    thisone->idx = (fff_indexed_data*)calloc(n, sizeof(fff_indexed_data));

  return thisone;  
}

static void _fff_onesample_mfx_delete(fff_onesample_mfx* thisone)
{

  fff_vector_delete(thisone->w); 
  fff_vector_delete(thisone->z); 
  fff_matrix_delete(thisone->Q); 
  fff_vector_delete(thisone->tvar);
  fff_vector_delete(thisone->tmp1); 
  fff_vector_delete(thisone->tmp2);
  if (thisone->idx != NULL) 
    free(thisone->idx); 

  free(thisone); 

  return; 
}



double fff_onesample_stat_mfx_eval(fff_onesample_stat_mfx* thisone, const fff_vector* x, const fff_vector* vx)
{
  double t; 
  t = thisone->compute_stat(thisone->params, x, vx, thisone->base); 
  return t; 
}



/*****************************************************************************************/
/*                   Standard MFX (normal population model)                              */
/*****************************************************************************************/
static double _fff_onesample_mean_gmfx(void* params, const fff_vector* x, const fff_vector* var, double base)
{
  unsigned int niter = *((unsigned int*)params); 
  double mu = 0.0, v = 0.0; 

  _fff_onesample_gmfx_EM(&mu, &v, x, var, niter, 0); 

  return (mu-base); 
}


static double _fff_onesample_LR_gmfx(void* params, const fff_vector* x, const fff_vector* var, double base)
{
  int sign; 
  double t, mu = 0.0, v = 0.0, v0 = 0.0, nll, nll0;
  unsigned int niter = *((unsigned int*)params); 
  
  /* Estimate maximum likelihood group mean and group variance */ 
  _fff_onesample_gmfx_EM(&mu, &v, x, var, niter, 0); 
  
  /* MFX mean estimate equals baseline, return zero */ 
  t = mu - base; 
  sign = FFF_SIGN(t); 
  if (sign == 0) 
    return 0.0; 

  /* Estimate maximum likelihood group variance under zero group mean assumption */ 
  _fff_onesample_gmfx_EM(&base, &v0, x, var, niter, 1); 

  /* Negated log-likelihoods */ 
  nll = _fff_onesample_gmfx_nll(x, var, mu, v);
  nll0 = _fff_onesample_gmfx_nll(x, var, base, v0);
  
  /* If both nll and nll0 are globally minimized, we always have: 
     nll0 >= nll; however, EM convergence issues may cause nll>nll0,
     in which case we return 0.0 */ 
  t = -2.0 * (nll - nll0);
  t = FFF_MAX(t, 0.0); 
  if (t < FFF_POSINF)
    return sign * sqrt(t); 
  /* To get perhaps a more "Student-like" statistic: 
     t = sign * sqrt((n-1)*(exp(t/nn) - 1.0)); */
  else if (sign > 0)
    return FFF_POSINF; 
  else 
    return FFF_NEGINF; 
}



/* EM algorithm to estimate the mean and variance parameters. */ 
static void _fff_onesample_gmfx_EM(double* m, double* v, 
				    const fff_vector* x, const fff_vector* var, 
				    unsigned int niter, int constraint)
{
  size_t n = x->size, i; 
  unsigned int iter = 0; 
  double nn=(double)n, m1, v1, m0, v0, mi_ap, vi_ap, aux; 
  double *bufx, *bufvar;
  
  /* Initialization: pure RFX solution (FFX variances set to zero) */ 
  if ( ! constraint ) 
    /** m1 = gsl_stats_mean(x->data, x->stride, n);
	v1 = gsl_stats_variance_with_fixed_mean(x->data, x->stride, n, m1); 
    **/
    v1 = fff_vector_ssd(x, &m1, 0)/(long double)x->size;  
  
  else {
    m1 = 0.0;
    v1 = fff_vector_ssd(x, &m1, 1)/(long double)x->size;  
  }

  /* Refine result using an EM loop */ 
  while (iter < niter) {

    /* Previous estimates */
    m0 = m1;
    v0 = v1;
    
    /* Loop: aggregated E- and M-steps */
    bufx = x->data; 
    bufvar = var->data;
    if ( ! constraint ) 
      m1 = 0.0; 
    v1 = 0.0; 
    for (i=0; i<n; i++, bufx+=x->stride, bufvar+=var->stride) {

      /* Posterior mean and variance of the true effect value */ 
      aux = 1.0 / (*bufvar + v0);
      mi_ap = v0 * (*bufx) + (*bufvar) * m0; 
      mi_ap *= aux; 
      vi_ap = aux * (*bufvar) * v0; 

      /* Update */ 
      if ( ! constraint )
	m1 += mi_ap; 
      v1 += vi_ap + FFF_SQR(mi_ap);
      
    }
    
    /* Normalization */
    if ( ! constraint ) 
      m1 /= nn; 
    v1 /= nn; 
    v1 -= FFF_SQR(m1); 
    
    /* Iteration number */ 
    iter ++; 
    
  }
  
  /* Save estimates */ 
  *m = m1; 
  *v = v1; 
  
  return;
}


/* Negated log-likelihood for the MFX model */ 
static double _fff_onesample_gmfx_nll(const fff_vector* x, const fff_vector* var, double m, double v)
{
  size_t n = x->size, i;
  double s, aux, ll = 0.0;
  double *bufx = x->data, *bufvar = var->data;
  
  for (i=0; i<n; i++, bufx+=x->stride, bufvar+=var->stride) {
    s = *bufvar + v;
    aux = *bufx - m;
    ll += log(s);
    ll += FFF_SQR(aux) / s;
  }
  
  ll *= .5;
  
  return ll;
}



/*****************************************************************************************/
/*                                  Empirical MFX                                        */
/*****************************************************************************************/
static double _fff_onesample_mean_mfx(void* params, const fff_vector* x, const fff_vector* var, double base)
{
  double m; 
  fff_onesample_mfx* Params = (fff_onesample_mfx*)params; 
  long double aux, sumw; 


  /* Estimate the population distribution using EM */ 
  _fff_onesample_mfx_EM(Params, x, var, 0);   

  /* Compute the mean of the estimated distribution */ 
  /** 
      m = gsl_stats_wmean (Params->w->data, Params->w->stride, Params->z->data, Params->z->stride, Params->z->size) - base; 
  **/
  aux = fff_vector_wsum(Params->z, Params->w, &sumw); 
  m = aux/sumw - base; 

  return m; 
}

static double _fff_onesample_median_mfx(void* params, const fff_vector* x, const fff_vector* var, double base)
{
  double m; 
  fff_onesample_mfx* Params = (fff_onesample_mfx*)params; 
  
  /* Estimate the population distribution using EM */ 
  _fff_onesample_mfx_EM(Params, x, var, 0);   
  
  /* Compute the median of the estimated distribution */ 
  /** m = fff_weighted_median(Params->idx, Params->w, Params->z) - base;  **/
  _fff_sort_z(Params->idx, Params->tmp1, Params->tmp2, Params->z, Params->w); 
  m = fff_vector_wmedian_from_sorted_data (Params->tmp1, Params->tmp2); 

  return m;
}



static double _fff_onesample_sign_stat_mfx(void* params, const fff_vector* x, const fff_vector* var, double base)
{
  fff_onesample_mfx* Params = (fff_onesample_mfx*)params; 
  double *buf, *bufw; 
  double aux, rp = 0.0, rm = 0.0;
  size_t i, n = x->size; 

  /* Estimate the population distribution using EM */ 
  _fff_onesample_mfx_EM(Params, x, var, 0);      

  /* Compute the sign statistic of the fitted distribution */ 
  buf = Params->z->data; 
  bufw = Params->w->data; 
  for (i=0; i<n; i++, buf+=Params->z->stride, bufw+=Params->w->stride) {
    aux = *buf - base; 
    if (aux > 0.0)
      rp += *bufw;
    else if (aux < 0.0) 
      rm += *bufw; 
    else { /* in case the center is exactly zero */ 
      aux = .5 * *bufw; 
      rp += aux; 
      rm += aux;
    }
  }
  
  return (rp-rm); 
}


static double _fff_onesample_wilcoxon_mfx(void* params, const fff_vector* x, const fff_vector* var, double base)
{
  double t = 0.0; 
  fff_onesample_mfx* Params = (fff_onesample_mfx*)params; 
  size_t i, n = x->size; 
  double *buf1, *buf2; 
  double zi, wi, Ri; 

  /* Estimate the population distribution using EM */ 
  _fff_onesample_mfx_EM(Params, x, var, 0);   

  /* Compute the vector of absolute residuals wrt the baseline */ 
  buf1 = Params->tmp1->data; 
  buf2 = Params->z->data; 
  for(i=0; i<n; i++, buf1+=Params->tmp1->stride, buf2+=Params->z->stride) {
    zi = *buf2 - base;
    *buf1 = FFF_ABS(zi); 
  }

  /* Sort the absolute residuals and get the permutation of indices */ 
  /**  gsl_sort_vector_index(Params->idx, Params->tmp1); **/
  _fff_sort_z(Params->idx, Params->tmp1, Params->tmp2, Params->z, Params->w); 
  
  /* Compute the sum of ranks */ 
  /** Ri = 0.0; 
  for(i=0; i<n; i++) {
    j = Params->idx->data[i];
    zi = Params->z->data[j*Params->z->stride]; 
    wi = Params->w->data[j*Params->w->stride]; 
    Ri += wi; 
    if (zi > base) 
      t += wi * Ri; 
    else if (zi < base)
      t -= wi * Ri; 
      }**/
  Ri = 0.0; 
  for(i=1, buf1=Params->tmp1->data, buf2=Params->tmp2->data; i<=n; i++) {
    zi = *buf1;
    wi = *buf2;
    Ri += wi; 
    if (zi > base) 
      t += wi * Ri; 
    else if (zi < base)
      t -= wi * Ri; 
  }

  return t; 
}


static double _fff_onesample_LR_mfx(void* params, const fff_vector* x, const fff_vector* var, double base)
{
  double t, mu, nll, nll0; 
  int sign; 
  fff_onesample_mfx* Params = (fff_onesample_mfx*)params; 
  long double aux, sumw; 

  /* Estimate the population distribution using EM */ 
  _fff_onesample_mfx_EM(Params, x, var, 0); 
  nll = _fff_onesample_mfx_nll(Params, x);

  /* Estimate the population mean */ 
  /**  mu = gsl_stats_wmean (Params->w->data, Params->w->stride, Params->z->data, Params->z->stride, Params->z->size); **/
  aux = fff_vector_wsum(Params->z, Params->w, &sumw); 
  mu = aux/sumw - base; 


  /* MFX mean estimate equals baseline, return zero */ 
  t = mu - base; 
  sign = FFF_SIGN(t); 
  if (sign == 0) 
    return 0.0; 

  /* Estimate the population distribution under zero mean constraint */ 
  _fff_onesample_mfx_EM(Params, x, var, 1); 
  nll0 = _fff_onesample_mfx_nll(Params, x);

  /* Compute the one-sided likelihood ratio statistic */ 
  t = -2.0 * (nll - nll0);
  t = FFF_MAX(t, 0.0); 
  if (t < FFF_POSINF)
    return sign * sqrt(t); 
  else if (sign > 0)
    return FFF_POSINF; 
  else 
    return FFF_NEGINF; 

}



/* EM algorithm to estimate the population distribution as a linear
   combination of Diracs centered at the datapoints. */ 

static void _fff_onesample_mfx_EM(fff_onesample_mfx* Params, 
				   const fff_vector* x, const fff_vector* var, 
				   int constraint)
{
  fff_vector *w = Params->w, *z = Params->z; 
  fff_vector *tvar = Params->tvar, *tmp1 = Params->tmp1, *tmp2 = Params->tmp2; 
  fff_matrix *Q = Params->Q; 
  unsigned int niter = *(Params->niter); 
  size_t n = x->size, i, k; 
  unsigned int iter = 0; 
  double m, lda, aux;
  double *buf, *buf2; 
  fff_vector Qk; 
  
  /* Pre-process: low threshold the variances to avoid numerical instabilities */ 
  aux = fff_vector_ssd(x, &m, 0)/(long double)(FFF_MAX(n,2)-1); 
  aux *= MIN_RELATIVE_VAR_FFX;
  fff_vector_memcpy(tvar, var); 
  buf = tvar->data; 
  for(i=0; i<n; i++, buf+=tvar->stride) {
    if (*buf < aux) 
      *buf = aux; 
  }
 
  /* Initial estimate: uniform weigths, class centers at datapoints */
  fff_vector_set_all(w, 1/(double)n); 
  fff_vector_memcpy(z, x); 

  /* Refine result using an EM loop */ 
  while (iter < niter) {
    
    /* Compute the posterior probability matrix 
       Qik : probability that subject i belongs to class k */ 
    _fff_onesample_mfx_EM_init(Params, x, 0); 
    
    /* Update weights: wk = sum_i Qik / n */ 
    buf = w->data; 
    for(k=0; k<n; k++, buf+=w->stride) {
      Qk = fff_matrix_col(Q, k); 
      *buf = fff_vector_sum(&Qk)/(long double)n; 
    }
    
    /* Reweight if restricted maximum likelihood: use the same Newton
       algorithm as in standard empirical likelihood */ 
    if ( constraint ) {
      fff_vector_memcpy(tmp1, z); 
      lda = _fff_el_solve_lda(tmp1, w); 
      if(lda < FFF_POSINF) {
	buf = z->data; 
	buf2 = w->data; 
	for(i=0; i<n; i++, buf+=z->stride, buf2+=w->stride) 
	  *buf2 *= 1/(1 + lda*(*buf)); 
      }
    }

    /* Update centers: zk = sum_i Rik xi  with Rik = Qik/si^2 */ 
    buf = z->data; 
    buf2 = tmp2->data; 
    for(k=0; k<n; k++, buf+=z->stride, buf2+=tmp2->stride) {
      
      /* Store the unconstrained ML update in z */ 
      Qk = fff_matrix_col(Q, k); 
      fff_vector_memcpy(tmp1, &Qk);       
      fff_vector_div(tmp1, tvar); /* Store Rik in tmp1 */ 
      aux = (double)fff_vector_sum(tmp1); /* aux == Rk = sum_i Rik */  
      aux = FFF_ENSURE_POSITIVE(aux);
      *buf = fff_blas_ddot(tmp1, x); /* z[k] = sum_i Rik xi */ 
      *buf /= aux;
      
      /* Store Rk = sum_i Rik in tmp2 */ 
      *buf2 = aux; 
          
    }


    /* Shift to zero if restricted maximum likelihood */
    if ( constraint ) {

      fff_vector_memcpy(tmp1, w); 
      fff_vector_div(tmp1, tmp2); /* tmp1_k == wk/Rk */ 
      
      aux = fff_blas_ddot(w, tmp1); /* aux == sum_k [ wk^2 / Rk ] */ 
      lda = fff_blas_ddot(w, z); /* lda = sum_k wk zk */ 

      aux = FFF_ENSURE_POSITIVE(aux);  
      lda /= aux; /* lda = sum_k wk zk /  sum_k [ wk^2 / Rk ] */ 
      
      fff_blas_daxpy(-lda, tmp1, z); /* zk = zk - lda * wk/Rk */ 
    }

    /* Iteration number */ 
    iter ++; 
    
  }
  
  return;
}


/* 
   If flag == 0, assemble the posterior probability matrix Q
      Qik : posterior probability that subject i belongs to class k. 
      Qik = ci wk g(xi-zk,si)
      ci determined by sum_k Qik = 1 
   
   Otherwise, assemble the likelihood matrix G 
      Gik = g(xi-zk,si)

*/ 
static void _fff_onesample_mfx_EM_init(fff_onesample_mfx* Params, 
					const fff_vector* x, int flag)
{
  fff_matrix* Q = Params->Q;
  const fff_vector *w = Params->w, *z = Params->z, *var = Params->tvar;
  size_t i, k, n = x->size, ii; 
  double xi, si; 
  double *bufQ, *bufxi, *bufvi, *bufwk, *bufzk;
  double sum = 0.0, aux; 

  /* Loop over subjects */ 
  bufxi = x->data; 
  bufvi = var->data; 
  for(i=0; i<n; i++, bufxi+=x->stride, bufvi+=var->stride) {
    
    xi = *bufxi;
    si = sqrt(*bufvi);
   
    ii = i*Q->tda; /* First element of the i-th line of Q */ 

    /* Loop over classes: compute Qik = wk * g(xi-zk,si), for each k */ 
    bufwk = w->data; 
    bufzk = z->data; 
    bufQ = Q->data + ii;
    sum = 0.0; 
    for(k=0; k<n; k++, bufQ++, bufwk+=w->stride, bufzk+=z->stride) {
      /** aux = gsl_ran_gaussian_pdf(xi-*bufzk, si); **/
      aux = (xi-*bufzk)/si; 
      aux = exp(-.5 * FFF_SQR(aux)); /* No need to divide by sqrt(2pi)si as it is constant */  
      *bufQ = FFF_ENSURE_POSITIVE(aux); /* Refrain posterior probabilities from vanishing */    
      if (flag == 0) {
	*bufQ *= *bufwk; 
	sum += *bufQ;
      }
    }

    /* Loop over classes: normalize Qik */ 
    if (flag == 0) {
      bufQ = Q->data + ii;
      for(k=0; k<n; k++, bufQ++) 
	*bufQ /= FFF_ENSURE_POSITIVE(sum);
    }
  
  }

  return; 
}


/* Negated empirical log-likelihood */ 
static double _fff_onesample_mfx_nll(fff_onesample_mfx* Params, 
				      const fff_vector* x)
			    
{
  const fff_vector *w = Params->w;
  fff_vector *Gw = Params->tmp1;
  fff_matrix* G = Params->Q;
  size_t i, n = w->size; 
  double aux, nll = 0.0; 
  double *buf;

  /* Compute G */ 
  _fff_onesample_mfx_EM_init(Params, x, 1);

  /* Compute Gw */ 
  fff_blas_dgemv(CblasNoTrans, 1.0, G, w, 0.0, Gw); 

  /* Compute the sum of logarithms of Gw */ 
  buf = Gw->data; 
  for (i=0; i<n; i++, buf+=Gw->stride) {
    aux = *buf; 
    aux = FFF_ENSURE_POSITIVE(aux); 
    nll -= log(aux); 
  }
  
  return nll; 
}



extern void fff_onesample_stat_mfx_pdf_fit(fff_vector* w, fff_vector* z, 
					   fff_onesample_stat_mfx* thisone, 
					   const fff_vector* x, const fff_vector* var)
{
  fff_onesample_mfx* Params = (fff_onesample_mfx*)thisone->params; 
  unsigned int constraint = thisone->constraint; 

  /* Check appropriate flag */ 
  if (!thisone->empirical)
    return; 

  /* Estimate the population distribution using EM */ 
  _fff_onesample_mfx_EM(Params, x, var, constraint);   

  /* Copy result in output vectors */ 
  fff_vector_memcpy(w, Params->w);
  fff_vector_memcpy(z, Params->z);

  return; 
}

  
extern void fff_onesample_stat_gmfx_pdf_fit(double *mu, double *v, 
					    fff_onesample_stat_mfx* thisone, 
					    const fff_vector* x, const fff_vector* var) 
{
  unsigned int niter = thisone->niter; 
  unsigned int constraint = thisone->constraint; 

  /* Estimate the population gaussian parameters using EM */ 
  _fff_onesample_gmfx_EM(mu, v, x, var, niter, constraint); 

}


/** Comparison function for qsort **/ 
static int _fff_indexed_data_comp(const void * x, const void * y)
{
  int ans = 1; 
  fff_indexed_data xx = *((fff_indexed_data*)x);
  fff_indexed_data yy = *((fff_indexed_data*)y);

  if (yy.x > xx.x) { 
    ans = -1; 
    return ans; 
  }
  if (yy.x == xx.x) 
    ans = 0; 

  return ans;  
}

/** Sort z array and re-order w accordingly **/ 
static void _fff_sort_z(fff_indexed_data* idx, fff_vector* tmp1, fff_vector* tmp2, 
			 const fff_vector* z, const fff_vector* w)
{
  size_t n = z->size, i, is; 
  double *buf1, *buf2; 
  fff_indexed_data* buf_idx; 

  /* Copy z into the auxiliary qsort structure idx */ 
  for(i=0, buf1=z->data, buf_idx=idx; 
       i<n; 
       i++, buf_idx++, buf1+=z->stride) {
    (*buf_idx).x = *buf1; 
    (*buf_idx).i = i; 
  }
  /* Effectively sort */
  qsort (idx, n, sizeof(fff_indexed_data), &_fff_indexed_data_comp);
  
  /* Copy the sorted z into tmp1, and the accordingly sorted w into tmp2 */ 
  for(i=0, buf1=tmp1->data, buf2=tmp2->data, buf_idx=idx; 
       i<n; 
       i++, buf_idx++, buf1+=tmp1->stride, buf2+=tmp2->stride) {
    is = (*buf_idx).i; 
    *buf1 = (*buf_idx).x; 
    *buf2 = w->data[ is*w->stride ]; 
  }

  return; 
}


/* Sign permutations */

void fff_onesample_permute_signs(fff_vector* xx, const fff_vector* x, double magic)
{
  size_t n = x->size, i; 
  double *bufx=x->data, *bufxx=xx->data; 
  double m = magic, aux; 

  for (i=0; i<n; i++, bufx+=x->stride, bufxx+=xx->stride) {
    aux = m/2; 
    m = FFF_FLOOR(aux); 
    aux -= m; 
    if (aux > 0) 
      *bufxx = -*bufx;
    else 
      *bufxx = *bufx; 
  }
  
  return; 
}
