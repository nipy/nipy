#include "fff_DPMM.h"
#include "fff_routines.h"
#include "fff_specfun.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <randomkit.h>


static int _recompute_and_redraw(fff_FDP* FDP, fff_array *Z, const fff_matrix *data, const fff_vector * pvals, const fff_array * labels, const int nit);

static int _withdraw (fff_FDP* FDP, fff_array *Z, const fff_matrix *data,  const fff_array * valid);

static int _compute_W(fff_matrix* W, const fff_FDP* FDP, const fff_matrix *data, const fff_vector * pvals, const fff_array * valid);

static double _theoretical_pval_gaussian(fff_vector * proba, const fff_vector * X,const fff_FDP* FDP);
static double _theoretical_pval_student(fff_vector * proba, const fff_vector * X,const fff_FDP* FDP);

static int _redraw(fff_array *Z, fff_matrix* W,const fff_array * valid, int nit);

static int _compute_P_under_H1(fff_vector *density, const fff_FDP* FDP, const fff_matrix *grid);

/**---------------------------------------**/

static int _recompute_and_redraw_IMM(fff_IMM* IMM, fff_array *Z, const fff_matrix *data, const fff_array * labels, const int nit);

static int _compute_P_IMM(fff_vector *density, const fff_IMM* IMM, const fff_matrix *grid);

static int _compute_W_IMM(fff_matrix* W, const fff_IMM* IMM, const fff_matrix *data, const fff_array * valid);

static double _pval_gaussian_(fff_vector * proba, const fff_vector * X,const fff_IMM* IMM);
static double _pval_WN_(fff_vector * proba, const fff_vector * X,const fff_IMM* IMM);

static int _withdraw_fixed (fff_IMM* IMM, fff_array *Z, const fff_matrix *data,  const fff_array * valid);

static int _withdraw_var (fff_IMM* IMM, fff_array *Z, const fff_matrix *data,  const fff_array * valid);

static int _withdraw_common (fff_IMM* IMM, fff_array *Z, const fff_matrix *data,  const fff_array * valid);

/************************************************************************/
/*********** Infinite Gaussian Mixture Model (IMM)  *********************/
/******************* with fixed covariance ******************************/
/************************************************************************/


/* "contructor" */
extern fff_IMM* fff_IMM_new( const double alpha, const long dim, const int type)
{
  fff_IMM* thisone;  

  /* Start with allocating the object */
  thisone = (fff_IMM*) calloc( 1, sizeof(fff_IMM) );
  
  /* Checks that the pointer has been allocated */
  if ( thisone == NULL) 
    return NULL; 
  
  int k=1;

  /* Initialization */
  thisone->alpha = alpha;
  thisone->dim = dim;
  thisone->k = k;
  thisone->type = type;
  thisone->prior_dof = 0;

  thisone->prior_means = fff_vector_new(dim);	
  thisone->prior_precisions = fff_vector_new(dim);
 thisone->prior_mean_scale = fff_vector_new(dim);

  thisone->means = fff_matrix_new(k,dim);
  thisone->prec_means = fff_matrix_new(k,dim);

  thisone->weights = fff_vector_new(k);  
  thisone->pop = fff_array_new1d(FFF_LONG,k);
  
  fff_vector_set(thisone->weights,0,alpha);

  if (thisone->type==1){
	thisone->precisions = fff_matrix_new(k,dim);
	thisone->dof = fff_vector_new(k); 
  }
  

  /* TODO : allocation tests */

  return thisone;  
}

/* "destructor" */
extern int fff_IMM_delete( fff_IMM* thisone )
{
  if ( thisone != NULL ) {
	fff_matrix_delete(thisone->means);
	fff_matrix_delete(thisone->prec_means);

	fff_vector_delete(thisone->weights);
	fff_array_delete(thisone->pop);

	fff_vector_delete(thisone->prior_precisions);
	fff_vector_delete(thisone->prior_means);
	fff_vector_delete(thisone->prior_mean_scale);
	if (thisone->type==1){
	  fff_vector_delete(thisone->dof);
	  fff_matrix_delete(thisone->precisions);
	}
	free(thisone);
  }
  return(0);
}

/* instantiate with priors */
extern int fff_fixed_IMM_instantiate( fff_IMM* thisone, const fff_vector *prior_precisions, const fff_vector* prior_means, const fff_vector* prior_mean_scale)
{
  fff_vector_memcpy(thisone->prior_precisions,prior_precisions);
  fff_vector_memcpy(thisone->prior_means, prior_means);
  fff_vector_memcpy(thisone->prior_mean_scale,prior_mean_scale);
  
  /* check the dimensions ...*/
  return(0);
}

extern int fff_var_IMM_instantiate( fff_IMM* thisone, const fff_vector *prior_precisions, const fff_vector* prior_means, const fff_vector* prior_mean_scale,const double prior_dof)
{
  fff_fixed_IMM_instantiate(thisone, prior_precisions, prior_means,prior_mean_scale);
  thisone ->prior_dof = prior_dof;
  return(0);
}



extern int fff_IMM_get_model( fff_matrix * mean, fff_matrix * prec_means, fff_vector * weights, const fff_IMM* IMM)
{
  /* To be updated; useful for fixed  variance only*/
  fff_vector_memcpy(weights, IMM->weights);
  fff_matrix_memcpy(mean, IMM->means);
  fff_matrix_memcpy(prec_means, IMM->prec_means);
  return IMM->k;
}

extern int fff_IMM_sampling(fff_vector *density, fff_IMM* IMM, fff_array *Z, const fff_matrix *data, const fff_array * labels, const fff_matrix *grid, const long niter)
{
  int i;

  fff_vector * W = fff_vector_new(grid->size1);

  for (i=0 ; i<niter;i++){
	_recompute_and_redraw_IMM(IMM,Z,data,labels,i);
	_compute_P_IMM(W, IMM, grid);
	fff_vector_add(density,W);
  }
  fff_vector_scale(density,1./niter);
  fff_vector_delete(W);
 
  return IMM->k;
}


extern int fff_IMM_estimation(fff_IMM* IMM, fff_array *Z, const fff_matrix *data, const fff_array * labels, const long niter)
{
  int i;
  int verbose = 0;
  fff_array_set_all(Z,0);

  for (i=0 ; i<niter;i++){	
	if (verbose) printf("%d %ld \n",i,IMM->k);
	_recompute_and_redraw_IMM(IMM,Z,data,labels,i); 
  }
  
  return IMM->k;
}

static int _recompute_and_redraw_IMM(fff_IMM* IMM,fff_array *Z, const fff_matrix *data, const fff_array * labels, const int nit)
{
  int i,j,s,S = (int) fff_array_max1d(labels)+1;
  fff_matrix *W ;
  fff_array * popl = fff_array_new1d(FFF_LONG,S);
  long aux;
  int n = labels->dimX;
  fff_array * valid = fff_array_new1d(FFF_LONG,n);

  for (i=0 ; i<n ; i++) {
	j = fff_array_get1d(labels,i);
	aux = fff_array_get1d(popl,j);
	fff_array_set1d(popl,j,aux+1);
  }
  
  for (s=0 ; s<S ; s++){
	for (i=0 ; i<n ; i++)
	  fff_array_set1d(valid,i,fff_array_get1d(labels,i)==s);
	if (fff_array_get1d(popl,s)>0){
	  if (IMM->type==0)
		_withdraw_fixed(IMM, Z, data, valid);
	  else
		_withdraw_var(IMM, Z, data, valid);
	  W = fff_matrix_new(n,IMM->k);
	  _compute_W_IMM(W, IMM, data,valid);
	  _redraw(Z,W,valid,nit);
	  fff_matrix_delete(W);
	}
  }
  
  fff_array_delete(popl);
  fff_array_delete(valid);
  return 0;
}

static int _compute_P_IMM(fff_vector *density, const fff_IMM* IMM, const fff_matrix *grid)
{
  int i;
  fff_vector * x = fff_vector_new(IMM->dim);
  fff_vector * w = fff_vector_new(IMM->k);
  double sw;

  for (i=0 ; i<grid->size1 ; i++) {
	fff_matrix_get_row (x, grid, i);
	sw = 0;
	if (IMM->type==0)
	  sw = _pval_gaussian_(w,x,IMM);
	else
	  sw = _pval_WN_(w,x,IMM);
	fff_vector_set(density,i,sw);
  }
  
  fff_vector_delete(x);
  fff_vector_delete(w);
  return 0;
}

static int _compute_W_IMM(fff_matrix* W, const fff_IMM* IMM, const fff_matrix *data, const fff_array * valid)
{
  int i;
  fff_vector * x = fff_vector_new(IMM->dim);
  fff_vector * w = fff_vector_new(IMM->k);

  for (i=0 ; i<valid->dimX ; i++) {
	if (fff_array_get1d(valid,i)==1){
	  fff_matrix_get_row (x, data, i);
	  if (IMM->type==0)
		_pval_gaussian_(w,x,IMM);
	  else
		_pval_WN_(w,x,IMM);
	  fff_matrix_set_row(W,i,w);
	}
  }
  fff_vector_delete(x);
  fff_vector_delete(w);
  return 0;
}

static double _pval_gaussian_(fff_vector * proba, const fff_vector * X,const fff_IMM* IMM)
{
  int i,j;
  double m,w,p,x,tau;
  double sw=0;

  for (i=0 ; i<IMM->k ; i++){
	w = 0;
	for (j=0 ; j<IMM->dim ; j++){
	  m = fff_matrix_get(IMM->means,i,j);
	  tau =  fff_vector_get(IMM->prior_mean_scale,j)+fff_array_get1d(IMM->pop,i);
	  tau = tau/(1+tau);
	  p = fff_vector_get(IMM->prior_precisions,j)*tau;
	  x = fff_vector_get(X,j);
	  w = w + log(p) - log(2*M_PI) - (m-x)*(m-x)*p; 
	}
	w = w/2;
	w = exp(w);
	w = w * fff_vector_get(IMM->weights,i);
	sw += w;
	fff_vector_set(proba,i,w);
  }
  return(sw);
}

static double _pval_WN_(fff_vector * proba, const fff_vector * X,const fff_IMM* IMM)
{
  int i,j;
  double m,w,x;
  double sw=0;

  double f,a,b,tau=0;

  for (i=0 ; i<IMM->k ; i++){
	w = 0;
	f = 0;
	a = fff_vector_get(IMM->dof,i);
	for (j=0 ; j<IMM->dim ; j++){	
	  tau = fff_vector_get(IMM->prior_mean_scale,j);
	  tau += fff_array_get1d(IMM->pop,i);
	  tau = tau/(1+tau);
	  m = fff_matrix_get(IMM->means,i,j);
	  b = fff_matrix_get(IMM->precisions,i,j);
	  x = fff_vector_get(X,j);
	  f = f+ log( 1./b + tau* (m-x)*(m-x));
	  w = w - log(b)*a;
	  w += 2*fff_gamln ((a+1-j)/2); /* NB  : might be incorrect due to diagonal model...*/
	  w -= 2*fff_gamln ((a-j)/2); /* NB  : might be incorrect due to diagonal model...*/
	}
	w = w -f*(a+1) + log(tau)*IMM->dim;
	w = w-log(M_PI)*IMM->dim;
	w = w/2;
	w = w + log(fff_vector_get(IMM->weights,i));

	w = exp(w);
	sw += w;
	fff_vector_set(proba,i,w);
  }

  return(sw);
}

static int _withdraw_common (fff_IMM* IMM, fff_array *Z, const fff_matrix *data,  const fff_array * valid)
{
  int l,i,m,j;
  double aux,sw,w;
  long laux;
  
  /* compute the population per cluster */
  fff_array_set_all(IMM->pop,0);
  for (i=0 ; i<valid->dimX ; i++) {
	if (fff_array_get1d(valid,i)==0){
	  l = fff_array_get1d(Z,i);
		laux = fff_array_get1d(IMM->pop,l);
		fff_array_set1d(IMM->pop,l,laux+1);
	}
  }

  /* update Z by removing empty components/clusters */
  fff_array * relabel = fff_array_new1d(FFF_LONG,IMM->k);
  int k=0;
  for (i=0 ; i<IMM->k ; i++)
	if (fff_array_get1d(IMM->pop,i)>0){
	  fff_array_set1d(relabel,i,k);
	  k++;
	}
  
  for (i=0 ; i<valid->dimX ; i++) {
	if (fff_array_get1d(valid,i)==0){
	  l = fff_array_get1d(Z,i);
		l = fff_array_get1d(relabel,l);
		fff_array_set1d(Z,i,l);
	}
  }
  
  /* reset pop */ 
  fff_array* pop = fff_array_new1d(FFF_LONG,k+1);
  fff_array_set_all(pop,0);
  for (i=0 ; i<IMM->k ; i++){
	long toto = fff_array_get1d(IMM->pop,i);
	if (toto>0)
	  fff_array_set1d(pop,fff_array_get1d(relabel,i),toto);
  }
  fff_array_delete(IMM->pop);
  fff_array_delete(relabel);
  IMM->pop = pop;
  IMM->k = k+1;
  
  /* compute the weights */
  fff_vector_delete(IMM->weights);
  IMM->weights = fff_vector_new(IMM->k);
  fff_vector_set(IMM->weights,IMM->k-1, IMM->alpha);
  sw = IMM->alpha;
  for (i=0 ; i<IMM->k-1 ; i++){
	w = (double) fff_array_get1d(IMM->pop,i);
	fff_vector_set(IMM->weights,i,w);
	sw +=w;
  }
  fff_vector_scale(IMM->weights,1./sw);
  
  /* compute the empirical means */
  
  fff_matrix *empmeans = fff_matrix_new(IMM->k, IMM->dim);
  for (i=0 ; i<valid->dimX ; i++) {
	if (fff_array_get1d(valid,i)==0){
	  l = fff_array_get1d(Z,i);
	  for (m=0 ; m<IMM->dim ; m++){
		aux =  fff_matrix_get(empmeans,l,m)+ fff_matrix_get(data,i,m);
		fff_matrix_set(empmeans,l,m,aux);
	  }
	}
  }

  /* compute the means */ 
  fff_matrix_delete(IMM->means);
  IMM->means = fff_matrix_new(IMM->k, IMM->dim);
  for (i=0 ; i<IMM->k ; i++){
	for (j=0 ; j<IMM->dim ; j++){
	  aux = fff_matrix_get(empmeans,i,j);
	  aux += fff_vector_get(IMM->prior_means,j)*fff_vector_get(IMM->prior_mean_scale,j);
	  aux /= (fff_array_get1d(IMM->pop,i)+fff_vector_get(IMM->prior_mean_scale,j));
	  fff_matrix_set(IMM->means,i,j,aux);
	}
  }
  fff_matrix_delete(empmeans);
  return IMM->k;
}

static int _withdraw_fixed (fff_IMM* IMM, fff_array *Z, const fff_matrix *data,  const fff_array * valid)
{
  
  int j,i;
  double aux;
  
  _withdraw_common (IMM,Z, data, valid);

  /* reset the precision on the mean  */
  double w;
  fff_matrix_delete(IMM->prec_means);
  IMM->prec_means = fff_matrix_new(IMM->k, IMM->dim);
  for (i=0 ; i<IMM->k ; i++){
	w = (double) fff_array_get1d(IMM->pop,i);
	for (j=0 ; j<IMM->dim ; j++){
	  aux = fff_vector_get(IMM->prior_precisions,j);
	  w += fff_vector_get(IMM->prior_mean_scale,j);
	  aux *=w;
	  fff_matrix_set(IMM->prec_means,i,j,aux);
	}
  }

 
  return IMM->k;
}

static int _withdraw_var (fff_IMM* IMM, fff_array *Z, const fff_matrix *data,  const fff_array * valid)
{
  
  int j,i,l,m;
  double aux;
  
  _withdraw_common (IMM,Z, data, valid);

  /* reset the dof */
  fff_vector_delete(IMM->dof);
  IMM->dof = fff_vector_new(IMM->k);
  fff_vector_set_all(IMM->dof,IMM->prior_dof);
  for (i=0 ; i<IMM->k ; i++)
	fff_vector_set(IMM->dof,i,fff_vector_get(IMM->dof,i)+fff_array_get1d(IMM->pop,i));

  /* compute the covariance */
  fff_matrix * emp_covariance = fff_matrix_new(IMM->k, IMM->dim);  
  for (i=0 ; i<valid->dimX ; i++) {
	if (fff_array_get1d(valid,i)==0){
	  l = fff_array_get1d(Z,i);
	  for (m=0 ; m<IMM->dim ; m++){
		aux = fff_matrix_get(data,i,m)-fff_matrix_get(IMM->means,l,m);
		aux =  fff_matrix_get(emp_covariance,l,m)+ aux*aux;
		fff_matrix_set(emp_covariance,l,m,aux);
	  }
	}
  }
   
  /* reset the precision */
  fff_matrix_delete(IMM->precisions);
  IMM->precisions = fff_matrix_new(IMM->k, IMM->dim);
  for (i=0 ; i<IMM->k ; i++){
	for (j=0 ; j<IMM->dim ; j++){
	  aux = fff_matrix_get(emp_covariance,i,j);
	  aux += 1/fff_vector_get(IMM->prior_precisions,j);
	  fff_matrix_set(IMM->precisions,i,j,1.0/aux);
	}
  }

  fff_matrix_delete(emp_covariance);
  
  /* reset the precision on the mean  */
  /* in this model, this is not necessary */

  
  return IMM->k;
}




/************************************************************************/
/********************** Fake DP  ****************************************/
/************************************************************************/


/* "contructor" */
extern fff_FDP* fff_FDP_new( const double alpha, const double g0, const double g1, const long dim, const double prior_dof)
{
  fff_FDP* thisone;  

  /* Start with allocating the object */
  thisone = (fff_FDP*) calloc( 1, sizeof(fff_FDP) );
  
  /* Checks that the pointer has been allocated */
  if ( thisone == NULL) 
    return NULL; 
  
  int k=2;

  /* Initialization */
  thisone->alpha = alpha;
  thisone->g0 = g0;
  thisone->g1 = g1;
  thisone->dim = dim;
  thisone->k = k;
  thisone->prior_dof = prior_dof;
  /* printf("Dof = %f \n",prior_dof); */

  thisone->means = fff_matrix_new(k-1,dim);
  thisone->precisions = fff_matrix_new(k-1,dim);
  thisone->prior_precisions = fff_matrix_new(1,dim);
  thisone->weights = fff_vector_new(k-1);  
  thisone->pop = fff_array_new1d(FFF_LONG,k);
  /* thisone->empmeans = fff_matrix_new(k-1,dim); */

  fff_vector_set(thisone->weights,0,alpha);

  /* TODO : allocation tests */

  return thisone;  
}

/* "destructor" */
extern int fff_FDP_delete( fff_FDP* thisone )
{
  if ( thisone != NULL ) {
	fff_matrix_delete(thisone->means);
	fff_vector_delete(thisone->weights);
	fff_matrix_delete(thisone->precisions);
	fff_array_delete(thisone->pop);
	/* fff_matrix_delete(thisone->empmeans); */

	free(thisone);
  }
  return(0);
}

/* instantiate with priors */
extern int fff_FDP_instantiate( fff_FDP* thisone, const fff_matrix *precisions )
{
  fff_matrix_memcpy(thisone->prior_precisions,precisions);
  /* check the dimensions ...*/
  return(0);
}

extern int fff_FDP_get_model( fff_matrix * mean, fff_matrix * precision, fff_vector * weights, const fff_FDP* FDP)
{
  fff_vector_memcpy(weights, FDP->weights);
  fff_matrix_memcpy(mean, FDP->means);
  fff_matrix_memcpy(precision, FDP->precisions);
  return FDP->k;
}

extern int fff_FDP_sampling(fff_vector *density, fff_FDP* FDP, fff_array *Z, const fff_matrix *data, const fff_vector * pvals, const fff_array * labels, const fff_matrix *grid, const long niter)
{
  int i;

  fff_vector * W = fff_vector_new(grid->size1);

  for (i=0 ; i<niter;i++){
	_recompute_and_redraw(FDP,Z,data,pvals,labels,i);
	_compute_P_under_H1(W, FDP, grid);
	fff_vector_add(density,W);
  }
  fff_vector_scale(density,1./niter);
  fff_vector_delete(W);
 
  return FDP->k;
}

extern int fff_FDP_inference(fff_FDP* FDP, fff_array *Z, fff_vector* posterior, const fff_matrix *data, const fff_vector * pvals, const fff_array * labels, const long niter)
{
  int i,n;
  double aux;
  fff_vector_set_all(posterior,0);

  for (i=0 ; i<niter;i++){
	_recompute_and_redraw(FDP,Z,data,pvals,labels,i); 
	for (n=0 ; n<data->size1 ; n++){
	  aux = (fff_array_get1d(Z,n)>0) + fff_vector_get(posterior,n);
	  fff_vector_set(posterior,n,aux);
	}
  }
  fff_vector_scale(posterior,1./niter);

  return FDP->k;
}

extern int fff_FDP_estimation(fff_FDP* FDP, fff_array *Z, const fff_matrix *data, const fff_vector * pvals, const fff_array * labels, const long niter)
{
  int i;
  fff_array_set_all(Z,-1);

  for (i=0 ; i<niter;i++){
	_recompute_and_redraw(FDP,Z,data,pvals,labels,i); 
  }
  return FDP->k;
}



static int _recompute_and_redraw(fff_FDP* FDP,fff_array *Z, const fff_matrix *data, const fff_vector * pvals, const fff_array * labels, const int nit)
{
  int i,j,s,S = (int) fff_array_max1d(labels)+1;
  fff_matrix *W ;
  fff_array * popl = fff_array_new1d(FFF_LONG,S);
  long aux;
  int n = labels->dimX;
  fff_array * valid = fff_array_new1d(FFF_LONG,n);

  for (i=0 ; i<n ; i++) {
	j = fff_array_get1d(labels,i);
	aux = fff_array_get1d(popl,j);
	fff_array_set1d(popl,j,aux+1);
  }
  
  for (s=0 ; s<S ; s++){
	for (i=0 ; i<n ; i++)
	  fff_array_set1d(valid,i,fff_array_get1d(labels,i)==s);
	_withdraw (FDP, Z, data, valid);
	if (fff_array_get1d(popl,s)>0){
	  W = fff_matrix_new(n,FDP->k);
	  _compute_W(W, FDP, data, pvals, valid);
	  _redraw(Z,W,valid,nit);
	  fff_matrix_delete(W);
	}
	
  }
  
  fff_array_delete(popl);
  fff_array_delete(valid);
  return 0;
}

static int _withdraw (fff_FDP* FDP, fff_array *Z, const fff_matrix *data,  const fff_array * valid)
{
  
  int j,l,i,m;
  double aux,temp;
  long laux;
  int n = valid->dimX;
 
  /* compute the population */
  fff_array_set_all(FDP->pop,0);
  for (i=0 ; i<n ; i++) {
	if (fff_array_get1d(valid,i)==0){
	  l = fff_array_get1d(Z,i);
	  if (l>-1){
		laux = fff_array_get1d(FDP->pop,l);
		fff_array_set1d(FDP->pop,l,laux+1);
	  }
	}
  }

  /* update Z by removing null components  */
  fff_array * relabel = fff_array_new1d(FFF_LONG,FDP->k);
  int k=1;
  for (i=1 ; i<FDP->k ; i++)
	if (fff_array_get1d(FDP->pop,i)>0){
	  fff_array_set1d(relabel,i,k);
	  k++;
	}

  FDP->k = k+1;
  for (i=0 ; i<n ; i++) {
	if (fff_array_get1d(valid,i)==0){
	  l = fff_array_get1d(Z,i);
	  if (l>0){
		l = fff_array_get1d(relabel,l);
		fff_array_set1d(Z,i,l);
	  }
	}
  }
  fff_array_delete(relabel);
  
  /* reset pop */
  fff_array_delete(FDP->pop);
  FDP->pop = fff_array_new1d(FFF_LONG,FDP->k);
  fff_array_set_all(FDP->pop,0);
  for (i=0 ; i<n ; i++) {
	if (fff_array_get1d(valid,i)==0){
	  l = fff_array_get1d(Z,i);
	  if (l>-1){
		laux = fff_array_get1d(FDP->pop,l);
		fff_array_set1d(FDP->pop,l,laux+1);
	  }
	}
  }
  
  /* compute the empirical means and covariance */
  /* fff_matrix_delete(FDP->empmeans); */
  fff_matrix *empmeans = fff_matrix_new(FDP->k-1, FDP->dim);
  fff_matrix *empcovar = fff_matrix_new(FDP->k-1, FDP->dim);
  for (i=0 ; i<n ; i++) {
	if (fff_array_get1d(valid,i)==0){
	  l = fff_array_get1d(Z,i);
	  if (l>0)
		for (m=0 ; m<FDP->dim ; m++){
		  temp = fff_matrix_get(data,i,m);
		  aux =  fff_matrix_get(empmeans,l-1,m)+ temp;
		  fff_matrix_set(empmeans,l-1,m,aux);
		  aux =  fff_matrix_get(empcovar,l-1,m)+ temp*temp;
		  fff_matrix_set(empcovar,l-1,m,aux);
		}
	}
  }
  
  /* compute the weights */
  fff_vector_delete(FDP->weights);
  FDP->weights = fff_vector_new(FDP->k-1);
  fff_vector_set_all(FDP->weights, FDP->alpha);
  double w,sw = FDP->alpha;
  for (i=1 ; i<FDP->k-1 ; i++){
	w = (double) fff_array_get1d(FDP->pop,i);
	fff_vector_set(FDP->weights,i-1,w);
	sw +=w;
  }
  fff_vector_scale(FDP->weights,1./sw);
  
  /* compute the means */
  if (k>2){
	fff_matrix_delete(FDP->means);
	FDP->means = fff_matrix_new(FDP->k-2, FDP->dim);
	for (i=0 ; i<FDP->k-2 ; i++){
	  w = (double)fff_array_get1d(FDP->pop,i+1);
	  for (j=0 ; j<FDP->dim ; j++){
		aux = fff_matrix_get(empmeans,i,j);
		aux /=w;
		fff_matrix_set(FDP->means,i,j,aux);
		temp = fff_matrix_get(empcovar,i,j)-aux*aux*w;
		fff_matrix_set(empcovar,i,j,temp);
	  }
	}
  }
  

  /* reset the precision */
  fff_matrix * precision;
  if (FDP->k>2){	
	precision = fff_matrix_new(FDP->k-2,FDP->dim);
	if (FDP->prior_dof==0){ 
	  /* this means infinite prior dof (!), thus posterior=prior */
	  for (i=0 ; i<FDP->k-2 ; i++)
		for (m=0 ; m<FDP->dim ; m++){
		  aux = fff_matrix_get(FDP->prior_precisions,0,m);
		  fff_matrix_set(precision,i,m,aux);
		}
	}
	else{/* finite prior_dofs */
	  for (i=0 ; i<FDP->k-2 ; i++)
		for (m=0 ; m<FDP->dim ; m++){
		  aux = FDP->prior_dof*1.0/fff_matrix_get(FDP->prior_precisions,0,m);
		  aux += fff_matrix_get(empcovar,i,m);
		  fff_matrix_set(precision,i,m,1.0/aux);
		}
	}
	fff_matrix_delete (FDP->precisions);
	FDP->precisions = precision; 
  }
  fff_matrix_delete(empmeans);
  fff_matrix_delete(empcovar);
  return FDP->k;
}

static int _compute_W(fff_matrix* W, const fff_FDP* FDP, const fff_matrix *data, const fff_vector * pvals, const fff_array * valid)
{
  int i,k;
  double p0;
  fff_vector * x = fff_vector_new(FDP->dim);
  fff_vector * w = fff_vector_new(FDP->k);

  for (i=0 ; i<valid->dimX ; i++) {
	if (fff_array_get1d(valid,i)==1){
	  p0 = 1-fff_vector_get(pvals,i);
	  fff_matrix_set(W,i,0,p0*FDP->g0);
	  fff_matrix_get_row (x, data, i);
	  if (FDP->prior_dof==0)
		_theoretical_pval_gaussian(w,x,FDP);
	  else
		_theoretical_pval_student(w,x,FDP);
	  for (k=0 ; k<FDP->k-1; k++)
		fff_matrix_set(W,i,k+1,(1-p0)*fff_vector_get(w,k));
	}
  }
  fff_vector_delete(x);
  fff_vector_delete(w);
  return 0;
}

static int _compute_P_under_H1(fff_vector *density, const fff_FDP* FDP, const fff_matrix *grid)
{
  int i;
  fff_vector * x = fff_vector_new(FDP->dim);
  fff_vector * w = fff_vector_new(FDP->k);
  double sw;

  for (i=0 ; i<grid->size1 ; i++) {
	fff_matrix_get_row (x, grid, i);
	if (FDP->prior_dof==0)
	  sw = _theoretical_pval_gaussian(w,x,FDP);
	else
	  sw = _theoretical_pval_student(w,x,FDP);
	fff_vector_set(density,i,sw);
  }
  
  fff_vector_delete(x);
  fff_vector_delete(w);
  return 0;
}

static double _theoretical_pval_student(fff_vector * proba, const fff_vector * X,const fff_FDP* FDP)
{
  int i,j;
  double m,w,x,a ,tau,f,b;
  double sw=0;
  
  for (i=0 ; i<FDP->k-2 ; i++){
	w = 0;
	f = 0;
	tau = 0;
	a = FDP->prior_dof + fff_array_get1d(FDP->pop,i);/* ? */
	tau = fff_array_get1d(FDP->pop,i);
	tau = tau/(1+tau);tau=1;
	for (j=0 ; j<FDP->dim ; j++){  
	  m = fff_matrix_get(FDP->means,i,j);
	  b = fff_matrix_get(FDP->precisions,i,j);
	  x = fff_vector_get(X,j);
	  f = f+ log( 1./b + tau* (m-x)*(m-x));
	  w = w - log(b)*a;
	  w += 2*fff_gamln ((a+1-j)/2);/* NB  : might be incorrect for diagonal model... */
	  w -= 2*fff_gamln ((a-j)/2);/* NB  : might be incorrect for diagonal model... */ 
	}
	w = w -f*(a+1) + log(tau)*FDP->dim;
	w = w-log(M_PI)*FDP->dim;
	
	w = w/2;
	w = exp(w);
	fff_vector_set(proba,i,w);
  }
  fff_vector_set(proba,FDP->k-2,FDP->g1);

  for (i=0 ; i<FDP->k-1 ; i++){
	w = fff_vector_get(proba,i);
	w = w * fff_vector_get(FDP->weights,i);
	sw += w;
	fff_vector_set(proba,i,w);
  }
  
  return(sw);
}

static double _theoretical_pval_gaussian(fff_vector * proba, const fff_vector * X,const fff_FDP* FDP)
{
  int i,j;
  double m,w,p,x;
  double sw=0;

  for (i=0 ; i<FDP->k-2 ; i++){
	w = 0;
	for (j=0 ; j<FDP->dim ; j++){
	  m = fff_matrix_get(FDP->means,i,j);
	  p = fff_matrix_get(FDP->precisions,i,j);
	  x = fff_vector_get(X,j);
	  w = w + log(p) - log(2*M_PI) - (m-x)*(m-x)*p; 
	}
	w = w/2;
	w = exp(w);
	fff_vector_set(proba,i,w);
  }
  fff_vector_set(proba,FDP->k-2,FDP->g1);

  for (i=0 ; i<FDP->k-1 ; i++){
	w = fff_vector_get(proba,i);
	w = w * fff_vector_get(FDP->weights,i);
	sw += w;
	fff_vector_set(proba,i,w);
  }
  
  return(sw);
}

static int _redraw(fff_array *Z, fff_matrix* W,const fff_array * valid, int nit)
{
  int n,j; 
   rk_state state; 
  rk_seed(nit, &state);
  double sp,h;

  for (n=0 ; n<valid->dimX ; n++) {
	if (fff_array_get1d(valid,n)==1){
	  sp = 0;
	  for (j=0 ; j<W->size2 ; j++)
		sp += fff_matrix_get(W,n,j);
	  
	  h = rk_double(&state)*sp;
	  sp = 0;
	  for (j=0 ; j<W->size2 ; j++){
		sp +=fff_matrix_get(W,n,j);
		if (sp>h) break;
	  }
	  j = FFF_MIN(j,W->size2-1);
	  fff_array_set1d(Z,n,j);
	}
  }
  return 0;
}
