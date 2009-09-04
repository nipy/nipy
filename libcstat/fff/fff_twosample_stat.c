#include "fff_twosample_stat.h"
#include "fff_onesample_stat.h"
#include "fff_gen_stats.h"
#include "fff_glm_twolevel.h"
#include "fff_base.h"

#include <stdlib.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>



static double _fff_twosample_student(void* params, const fff_vector* x, unsigned int n1);
static double _fff_twosample_wilcoxon(void* params, const fff_vector* x, unsigned int n1);
static double _fff_twosample_student_mfx(void* params, const fff_vector* x, 
					 const fff_vector* vx, unsigned int n1);
static void  _fff_twosample_mfx_assembly(fff_matrix* X, fff_matrix* PX, fff_matrix* PPX, 
					 unsigned int n1, unsigned int n2);


typedef struct{
  fff_glm_twolevel_EM *em; 
  unsigned int* niter; 
  fff_vector* work; 
  fff_matrix* X;
  fff_matrix* PX; 
  fff_matrix* PPX; 
} fff_twosample_mfx; 


fff_twosample_stat* fff_twosample_stat_new(unsigned int n1, unsigned int n2, 
					   fff_twosample_stat_flag flag)
{
  fff_twosample_stat* thisone = (fff_twosample_stat*)malloc(sizeof(fff_twosample_stat)); 

  if (thisone == NULL) {
    FFF_ERROR("Cannot allocate memory", ENOMEM); 
    return NULL; 
  }

  thisone->n1 = n1; 
  thisone->n2 = n2; 
  thisone->flag = flag; 
  thisone->params = NULL; 

  switch (flag) {

  case FFF_TWOSAMPLE_STUDENT:
    thisone->compute_stat = &_fff_twosample_student;
    break;
    
  case FFF_TWOSAMPLE_WILCOXON:
    thisone->compute_stat = &_fff_twosample_wilcoxon;
    break; 

  default:
    FFF_ERROR("Unrecognized statistic", EINVAL);
    break; 
  }

  return thisone; 
}
  

void fff_twosample_stat_delete(fff_twosample_stat* thisone)
{
  if (thisone == NULL) 
    return; 
  free(thisone); 
  return; 
}


double fff_twosample_stat_eval(fff_twosample_stat* thisone, const fff_vector* x)
{
  double t; 
  t = thisone->compute_stat(thisone->params, x, thisone->n1);
  return t; 
}


fff_twosample_stat_mfx* fff_twosample_stat_mfx_new(unsigned int n1, unsigned int n2, 
						   fff_twosample_stat_flag flag)
{
  fff_twosample_stat_mfx* thisone = (fff_twosample_stat_mfx*)malloc(sizeof(fff_twosample_stat_mfx)); 
  fff_twosample_mfx* aux; 
  unsigned int n = n1+n2; 

  if (thisone == NULL) {
    FFF_ERROR("Cannot allocate memory", ENOMEM); 
    return NULL; 
  }
  
  thisone->n1 = n1; 
  thisone->n2 = n2; 
  thisone->flag = flag; 
  thisone->niter = 0; 

  switch (flag) {

  case FFF_TWOSAMPLE_STUDENT_MFX:
    thisone->compute_stat = &_fff_twosample_student_mfx;
    aux = (fff_twosample_mfx*)malloc(sizeof(fff_twosample_mfx)); 
    thisone->params = (void*)aux; 
    aux->em = fff_glm_twolevel_EM_new(n, 2);
    aux->niter = &(thisone->niter); 
    aux->work = fff_vector_new(n); 
    aux->X = fff_matrix_new(n, 2); 
    aux->PX = fff_matrix_new(2, n); 
    aux->PPX = fff_matrix_new(2, n); 
    _fff_twosample_mfx_assembly(aux->X, aux->PX, aux->PPX, n1, n2); 
    break;

  default:
    FFF_ERROR("Unrecognized statistic", EINVAL);
    break; 
  }

  return thisone; 
}


void fff_twosample_stat_mfx_delete(fff_twosample_stat_mfx* thisone)
{
  fff_twosample_mfx* aux; 

  if (thisone == NULL) 
    return; 
  
  switch (thisone->flag) {

  case FFF_TWOSAMPLE_STUDENT_MFX:
    aux = (fff_twosample_mfx*) thisone->params; 
    fff_vector_delete(aux->work);
    fff_matrix_delete(aux->X); 
    fff_matrix_delete(aux->PX); 
    fff_matrix_delete(aux->PPX); 
    fff_glm_twolevel_EM_delete(aux->em);
    free(aux);
    break;
    
  default:
    FFF_ERROR("Unrecognized statistic", EINVAL);
    break; 
  }
  free(thisone); 
  return; 
}

double fff_twosample_stat_mfx_eval(fff_twosample_stat_mfx* thisone, 
				   const fff_vector* x, const fff_vector* vx)
{
  double t; 
  t = thisone->compute_stat(thisone->params, x, vx, thisone->n1);
  return t; 
}



/*********************************************************************
            Actual test statistic implementation
**********************************************************************/

static double _fff_twosample_student(void* params, const fff_vector* x, unsigned int n1)
{
  fff_vector x1, x2;
  unsigned int naux = x->size-n1;
  double t, m1, m2; 
  long double v1, aux; 

  /* Compute within-group means and variances */
  x1 = fff_vector_view(x->data, n1, x->stride); 
  x2 = fff_vector_view(x->data+n1, naux, x->stride); 
  v1 = fff_vector_ssd(&x1, &m1, 0); 
  aux = fff_vector_ssd(&x2, &m2, 0);
  
  /* Compute max( n1+n2-2, 1 ) */ 
  naux += n1-2; 
  if (naux<=0)
    naux = 1; 

  /* Compute the inverse std estimate */ 
  aux += v1; 
  aux /= naux; 
  aux = sqrt(aux); 
  if (aux<=0.0) 
    aux = FFF_POSINF; 
  else 
    aux = 1/aux; 

  /* t value */ 
  t = (m1-m2)*aux;

  return t; 
}

/*
  Wilcoxon. 
*/
static double _fff_twosample_wilcoxon(void* params, const fff_vector* x, unsigned int n1)
{
  fff_vector x1, x2;
  unsigned int i, j, n2=x->size-n1;
  double w=0.0, aux; 
  double *b1, *b2; 

  x1 = fff_vector_view(x->data, n1, x->stride); 
  x2 = fff_vector_view(x->data+n1, n2, x->stride); 
   
  for(i=0, b1=x1.data; i<n1; i++, b1+=x1.stride) {
    aux = 0.0; 
    for(j=0, b2=x2.data; j<n2; j++, b2+=x2.stride) {
      if (*b1 > *b2) 
	aux += 1.0; 
      else if (*b2 > *b1)
	aux -= 1.0;      
    }
    aux /= (double)n2; 
    w += aux; 
  }

  return w; 
}



/*
  Pre-compute matrices for two-sample mixed-effect linear analysis. 

  X has two columns: c0 = [1 1 ... 1]' and c1 = [1 ... 1 | 0 ... 0]'

  

*/ 

static void  _fff_twosample_mfx_assembly(fff_matrix* X, fff_matrix* PX, fff_matrix* PPX, 
					 unsigned int n1, unsigned int n2)
{
  unsigned int n = n1+n2; 
  double g1=1/(double)n1, g2=1/(double)n2; 
  fff_matrix B; 
  
  /* X */ 
  fff_matrix_set_all(X, 1.0); 
  B = fff_matrix_block(X, n1, n2, 1, 1); 
  fff_matrix_set_all(&B, 0.0);  

  /* PX */ 
  B = fff_matrix_block(PX, 0, 1, 0, n1);
  fff_matrix_set_all(&B, 0.0);  
  B = fff_matrix_block(PX, 0, 1, n1, n2);
  fff_matrix_set_all(&B, g2);  
  B = fff_matrix_block(PX, 1, 1, 0, n1);
  fff_matrix_set_all(&B, g1);  
  B = fff_matrix_block(PX, 1, 1, n1, n2);
  fff_matrix_set_all(&B, -g2);  

  /* PPX */ 
  B = fff_matrix_block(PPX, 0, 1, 0, n);
  fff_matrix_set_all(&B, 1.0/(double)n);  
  B = fff_matrix_block(PPX, 1, 1, 0, n);
  fff_matrix_set_all(&B, 0.0);  

  return; 
}

static double _fff_twosample_student_mfx(void* params, const fff_vector* x, 
					 const fff_vector* vx, unsigned int n1)
{
  fff_twosample_mfx* Params = (fff_twosample_mfx*)params;
  double F, sign, ll, ll0; 
  unsigned int niter = *(Params->niter); 

  /* Constrained EM */ 
  fff_glm_twolevel_EM_init(Params->em);
  fff_glm_twolevel_EM_run(Params->em, x, vx, Params->X, Params->PPX, niter);
  ll0 = fff_glm_twolevel_log_likelihood(x, vx, Params->X, Params->em->b, Params->em->s2, Params->work);   

  /* Unconstrained EM initialized with constrained maximization results */ 
  fff_glm_twolevel_EM_run(Params->em, x, vx, Params->X, Params->PX, niter);
  ll = fff_glm_twolevel_log_likelihood(x, vx, Params->X, Params->em->b, Params->em->s2, Params->work);   

  /* Form the generalized F statistic */ 
  F = 2.0*(ll-ll0); 
  F = FFF_MAX(F, 0.0); /* Just to make sure */ 

  sign = Params->em->b->data[1]; /* Contiguity ensured */
  sign = FFF_SIGN(sign); 
  
  return sign*sqrt(F); 
}




/*********************************************************************
              Permutations
**********************************************************************/

unsigned int fff_twosample_permutation(unsigned int* idx1, unsigned int* idx2, 
				       unsigned int n1, unsigned int n2, double* magic)
{
  unsigned int n=FFF_MIN(n1, n2), i;
  double aux, magic1, magic2, cuml=0, cumr=1,c1=1, c2=1; 

  /* Pre-computation mode */
  if ( (idx1==NULL) || (idx2==NULL) ) 
    *magic = FFF_POSINF; 

  /* Find i such that Cn1,i*Cn2,i <= magic < Cn1,i*Cn2,i + Cn1,i+1*Cn2,i+1 */
  for(i=0; i<=n; i++) { 
    
    /* Downshift the magic number on exit */ 
    if (*magic<cumr) {
      *magic -= cuml; 
      break;
    }

    /* Compute Cn1,i+1 * Cn2,i+1 */
    aux = (double)(i+1); 
    c1 *= (n1-i); 
    c1 /= aux;
    c2 *= (n2-i); 
    c2 /= aux;

    /* Update bounds */ 
    cuml = cumr; 
    cumr += c1*c2; 

  }

  /* In pre-computation mode, return the number of permutations */ 
  if (*magic >= cumr) { /* AR,27/2/09 modified without certainty from *magic > cumr */
    *magic = cumr; 
    return 0; 
  }
    

  /* 
     Compute magic numbers for within-group combinations.
     We use: magic = magic2*c1 + magic1
  */ 
  magic2 = floor(*magic/c1);
  magic1 = *magic - magic2*c1;  

  /* Find the underlying combinations */ 
  fff_combination(idx1, i, n1, magic1);
  fff_combination(idx2, i, n2, magic2);

  return i; 
}

/*
  px assumed allocated n1 + n2
*/

#define SWAP(a, b)				\
  aux = a;					\
  a = b;					\
  b = aux

void fff_twosample_apply_permutation(fff_vector* px, fff_vector* pv, 
				     const fff_vector* x1, const fff_vector* v1, 
				     const fff_vector* x2, const fff_vector* v2,
				     unsigned int i, 
				     const unsigned int* idx1, const unsigned int* idx2)
{
  unsigned int j;
  size_t i1, i2, n1=x1->size, n2=x2->size; 
  double aux; 
  double *bpx1, *bpx2;
  fff_vector px1, px2, pv1, pv2; 
  int flag_mfx = (pv!=NULL); 

  /* Copy input vectors into single output vector */ 
  px1 = fff_vector_view(px->data, n1, px->stride); 
  fff_vector_memcpy(&px1, x1);  
  px2 = fff_vector_view(px->data + n1, n2, px->stride); 
  fff_vector_memcpy(&px2, x2);
  
  if (flag_mfx) {
      pv1 = fff_vector_view(pv->data, n1, pv->stride); 
      fff_vector_memcpy(&pv1, v1);  
      pv2 = fff_vector_view(pv->data + n1, n2, pv->stride); 
      fff_vector_memcpy(&pv2, v2);
  }

  /* Exchange elements */ 
  for(j=0; j<i; j++) {
    i1 = idx1[j];
    i2 = idx2[j];
    bpx1 = px1.data + i1*px->stride;
    bpx2 = px2.data + i2*px->stride;   
    SWAP(*bpx1, *bpx2); 
    if (flag_mfx) {    
      bpx1 = pv1.data + i1*pv->stride; 
      bpx2 = pv2.data + i2*pv->stride; 
      SWAP(*bpx1, *bpx2); 
    }
  }
  
  return; 
} 


