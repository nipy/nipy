#include "fff_clustering.h"
#include "fff_lapack.h"
#include "fff_GMM.h"
#include "fff_routines.h"
#include <randomkit.h>
#include "fff_specfun.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>


static void _fff_clustering_subsample(fff_matrix* X_short, fff_array *Label_short, const fff_matrix* X, const fff_array *Label);

int _fff_GMM_init(fff_matrix* Centers,fff_matrix* Precision, fff_vector *Weights,fff_matrix* X);
int _fff_GMM_init_hard(fff_matrix* Centers,fff_matrix* Precision, fff_vector *Weights, const fff_matrix* X, const fff_array* Label);
double _fff_update_gmm(fff_matrix* Centers, fff_matrix* Precision,  fff_vector *Weights, const fff_matrix* X);
double _fff_update_gmm_diag(fff_matrix* Centers, fff_matrix* Precision,  fff_vector *Weights, const fff_matrix* X);
double _fff_update_gmm_diag_dev( fff_matrix* Centers, fff_matrix* Precision,  fff_vector *Weights, fff_array **pa, fff_array *vo, const fff_matrix* X);
double _fff_update_gmm_hom(fff_matrix* Centers, fff_matrix* Precision, const fff_matrix* X );
double _fff_gmm_partition(fff_array* Labels, const fff_matrix* X, const fff_matrix* Centers, const fff_matrix* Precision, const fff_vector* Weights );

/* VB-GMM */
static int _fff_VBGMM_init(fff_Bayesian_GMM* BGMM);
static double _fff_VB_update_gmm(const fff_matrix* X, fff_Bayesian_GMM* BGMM);
static int _fff_VB_gmm_MAP(fff_array *Label, const fff_matrix* X, const fff_Bayesian_GMM* BGMM);
static int _fff_VB_log_norm(fff_vector* log_norm_fact, const fff_Bayesian_GMM* BGMM);

/* Gibbs-Bayesian GMM */
static int _fff_BGMM_init(fff_Bayesian_GMM* BG);
static double _fff_WNpval_(fff_vector * proba, const fff_vector *X, const fff_Bayesian_GMM* BG);
static double _fff_Npval(fff_matrix * proba, const fff_matrix *X, const fff_Bayesian_GMM* BG);

static double _fff_update_BGMM(fff_Bayesian_GMM* BG, const fff_matrix *X, int nit,const int method);

static double _fff_full_update_BGMM(fff_matrix * proba, fff_Bayesian_GMM* BG, const fff_matrix *X, int nit, const int method);

/* these ones should and will be put elsewhere */
/*
 static int _fff_LU_invert(fff_matrix *iM, fff_matrix *M);
// static double _fff_LU_det(fff_matrix *M);
//static int _generate_normals(fff_matrix* nvariate, const fff_matrix * mean, const fff_matrix * precision);
*/

/**********************************************************************
********************* Auxiliary function ******************************
**********************************************************************/




static void _fff_clustering_subsample(fff_matrix* X_short, fff_array *Label_short, const fff_matrix* X, const fff_array *Label)
{
  
  int N,i,n,fd;
  N = X->size1;
  n = X_short->size1;
  fd = X->size2;
  
  size_t* list = (size_t*) calloc( n,sizeof(size_t));
  if (!list) return;
  fff_vector * v = fff_vector_new(fd); 
  
  fff_rng_draw_noreplace (list, n, N);
  
  for (i=0 ; i<n ; i++){
    fff_array_set1d(Label_short,i,fff_array_get1d(Label,list[i])); 
    fff_matrix_get_row(v,X,list[i]);
    fff_matrix_set_row(X_short,i,v);
  }
  
  free(list);
  fff_vector_delete(v);
 
}


/**********************************************************************
********************* EM algorithm ******************************
**********************************************************************/

int fff_clustering_gmm_select( fff_matrix* Centers, fff_matrix* Precision,  fff_vector *Weights, fff_array *Label, const fff_matrix* X, const fff_vector *nbclust, const int maxiter, const double delta)
{
  char* proc = "fff_clustering_gmm_select";
  int i;
  double Li;
  double Lb = 0;
  int N = X->size1;
  int fd = X->size2;
  int fd2 = fd*fd;
  int k;

   int prec_type;
  
  if ((Precision->size1)==1)
    prec_type = 2;
  else
    if ((int)(Precision->size2)==fd2)
      prec_type = 0;
    else {
      if ((int)(Precision->size2)==fd)
	prec_type = 1;
      else return(0);
    }

  int ninit = nbclust->size;
  double *bufn = nbclust->data;
  int kb = 0;
  fff_matrix* Precision_aux = fff_matrix_new(Precision->size1, Precision->size2);
  fff_array* Label_aux = fff_array_new1d( FFF_LONG,N);
  fff_array* Label_init = fff_array_new1d( FFF_LONG,N);
  fff_array_copy(Label_init,Label);
  
  for (i=0 ; i<ninit ; i++, bufn++){
    k = (int)(*bufn);
    fff_matrix* Centers_aux = fff_matrix_new(k, fd);
    
    fff_vector* Weights_aux = fff_vector_new(k);
    fff_array_copy(Label_aux,Label_init);
    
    Li = fff_clustering_gmm(Centers_aux, Precision_aux, Weights_aux, Label_aux,X, maxiter, delta,N,0 );

    switch (prec_type) {
    case 0: /* full cluster-based covariance*/
      Li = Li-(k*fd*(fd+3)/2+k-1)*log(N)/(2*N);
      
    case 1: /*diagonal cluster-based covariance*/
      Li = Li-(k*fd*2+k-1)*log(N)/(2*N);
    case 2:/*diagonal average covariance, constant weights*/
      Li = Li-((k+1)*fd)*log(N)/(2*N);
    }
    
    if(i==0) Lb = Li-1;
    if (Li>Lb){
      kb = k;
      Lb = Li;
      fff_matrix_memcpy(Centers,Centers_aux);
      fff_matrix_memcpy(Precision,Precision_aux);
      fff_vector_memcpy(Weights,Weights_aux);
      fff_array_copy(Label, Label_aux); 
    }
    fff_matrix_delete(Centers_aux);    
    fff_vector_delete(Weights_aux);
    printf ("%s : %f %f %d\n",proc,Li,Lb,kb);
  }

  fff_matrix_delete(Precision_aux);
  fff_array_delete(Label_aux);
  fff_array_delete(Label_init);

  return(kb);
}

extern double fff_clustering_gmm_ninit( fff_matrix* Centers, fff_matrix* Precision,  fff_vector *Weights, fff_array *Label, const fff_matrix* X, const int maxiter, const double delta, const int ninit )
{
  /* char* proc = "fff_clustering_gmm_ninit"; */
  int i;
  double Li;
  double Lb = 0;
  int N = X->size1;
  int k = Centers->size1;
  int fd = X->size2;
  
  fff_matrix* Centers_aux = fff_matrix_new(k, fd);
  fff_matrix* Precision_aux = fff_matrix_new(Precision->size1, Precision->size2 );
  fff_vector* Weights_aux = fff_vector_new(k);
  fff_array* Label_aux = fff_array_new1d( FFF_LONG,N);
  fff_matrix_set_all( Centers,0 );
  fff_matrix_set_all( Precision,0 );
  fff_vector_set_all( Weights,0 );
  fff_array_set_all( Label,-1 );

  for (i=0; i<ninit; i++){
    Li = fff_clustering_gmm(Centers_aux, Precision_aux, Weights_aux, Label_aux, X, maxiter, delta, N ,0);
    
    if(i==0) Lb = Li-1;
    if (Li>Lb){
      fff_matrix_memcpy(Centers, Centers_aux );
      fff_matrix_memcpy(Precision, Precision_aux );
      fff_array_copy(Label, Label_aux);
      fff_vector_memcpy(Weights, Weights_aux);
    }
  }
  fff_matrix_delete(Centers_aux);
  fff_matrix_delete(Precision_aux);
  fff_vector_delete(Weights_aux);
  fff_array_delete(Label_aux);
  
  return(Lb);
} 

extern int fff_gmm_relax( fff_vector* LogLike, fff_array* Labels, fff_matrix* Centers, fff_matrix* Precision, fff_vector* Weights, const fff_matrix* X, const int maxiter, const double delta)
{
  char* proc = "fff_clustering_relax";
  int i;
  fff_vector* Like = fff_vector_new(maxiter);
  double L0 = 0;
  double La = 0;
  int fd = X->size2;
  int fd2 = fd*fd;
  int prec_type;
  int verbose = 0;
  
  if ((Precision->size1)==1)
    prec_type = 2;/*diagonal average covariance*/
  else
    if ((int)(Precision->size2)==fd2)
      prec_type = 0;/* full cluster-based covariance*/
    else {
      if ((int)(Precision->size2)==fd)
	prec_type = 1;/*diagonal cluster-based covariance*/
      else return(0);
    }
  
  for (i=0; i<maxiter; i++){
    switch (prec_type) {
    case 0:{
      Like->data[i] = _fff_update_gmm(Centers,Precision, Weights,X);
      break;
    }
    case 1:{
      Like->data[i] = _fff_update_gmm_diag(Centers,Precision, Weights,X);
	  break;
    }
    case 2:{
	  Like->data[i] = _fff_update_gmm_hom(Centers,Precision,X); 
	  break;
    }
    }
    if (verbose)
	  printf ("%s : it %d LL=%f\n",proc, i,Like->data[i]);
    if (i>0){
	  if (delta>Like->data[i]-La){
		La = Like->data[i];
		break;
      }
    }
    else
      L0 = Like->data[i];
    La = Like->data[i];
  }
    
  La = fff_gmm_partition(LogLike, Labels, X, Centers, Precision,Weights);

  fff_vector_delete( Like );
 
  return(La);
}

extern double fff_clustering_gmm( fff_matrix* Centers, fff_matrix* Precision,  fff_vector *Weights, fff_array *Label, const fff_matrix* X, const int maxiter, const double delta, const int chunksize, const int verbose )
{
  char* proc = "fff_clustering_gmm";
  int i;
  fff_vector* Like = fff_vector_new(maxiter);
  double L0 = 0;
  double La = 0;
  int fd = X->size2;
  int fd2 = fd*fd;
  int prec_type;
  int k = Centers->size1;
  
  if ((Precision->size1)==1)
    prec_type = 2;/*diagonal average covariance*/
  else
    if ((int)(Precision->size2)==fd2)
      prec_type = 0;/* full cluster-based covariance*/
    else {
      if ((int)(Precision->size2)==fd)
	prec_type = 1;/*diagonal cluster-based covariance*/
      else return(0);
    }
  
  fff_matrix* X_short;
  fff_array* Label_short;
  int N;
  N = X->size1;
  
  if (N>chunksize){
    
    Label_short =  fff_array_new1d( FFF_LONG,chunksize);
    X_short = fff_matrix_new( chunksize, fd); 
    
    _fff_clustering_subsample(X_short, Label_short, X, Label);
    
  }
  else{
    Label_short =  fff_array_new1d( FFF_LONG,N);
    X_short = fff_matrix_new(N, fd);
    fff_matrix_memcpy(X_short,X);
    fff_array_copy(Label_short,Label);
  } 
   
  if (fff_clustering_OntoLabel(Label_short,k)){
	_fff_GMM_init_hard(Centers,Precision,Weights,X_short,Label_short);
  }
  else{
	_fff_GMM_init(Centers,Precision,Weights,X_short);
  }
  
  fff_array *pa = fff_array_new1d(FFF_LONG,X->size1);
  fff_array *vo = fff_array_new1d(FFF_LONG,X->size1+1);
  for (i=0; i<maxiter; i++){
    switch (prec_type) {
    case 0:{
      Like->data[i] = _fff_update_gmm(Centers,Precision, Weights, X_short);
      break;
    }
    case 1:{
      Like->data[i] = _fff_update_gmm_diag(Centers,Precision, Weights,X_short);
      /* Like->data[i] = _fff_update_gmm_diag_dev(X_short,Centers,Precision, Weights,&pa,vo); */
		break;
    }
    case 2:{
	  Like->data[i] = _fff_update_gmm_hom(Centers,Precision,X_short); 
	  break;
    }
    }
    if (verbose)
	  printf ("%s it %d LL=%f\n",proc,i,Like->data[i]);
    if (i>0){
      /* if (delta*(La-L0)>Like->data[i]-La){ */ 
	  if (delta>Like->data[i]-La){
		La = Like->data[i];
		break;
	  }
    }
    else
      L0 = Like->data[i];
    La = Like->data[i];
  }
  
  
  La = _fff_gmm_partition(Label, X, Centers, Precision,Weights);

  fff_array_delete(pa);
  fff_array_delete(vo);
  fff_matrix_delete( X_short );
  fff_vector_delete( Like );
  fff_array_delete( Label_short );
 
  return(La);
}

int _fff_GMM_init(fff_matrix* Centers,fff_matrix* Precision, fff_vector *Weights,fff_matrix* X)
{
  /* char* proc = "_fff_gmm_init";*/
  int fd = X->size2;   
  int fd2 = fd*fd;
  int k = Centers->size1;
  int N = X->size1;
  int i,j,l,l2;
  size_t* seeds = (size_t*) calloc( k,sizeof(size_t));
  if (!seeds) return(0);
  double aux;
  fff_vector *v = fff_vector_new(fd);
  fff_vector *w = fff_vector_new(fd);

  /* init the weights */
  fff_vector_set_all(Weights, 1./k);
  
  /* init the centers */
  fff_rng_draw_noreplace(seeds, k, N);
  
  for (j=0 ; j<k ; j++){
    fff_matrix_get_row(v,X,seeds[j]);
    fff_matrix_set_row(Centers,j,v);
  }
  
  /* Compute the average signal */
  fff_vector_set_all( v,0 );

  for (i=0 ; i<N ; i++){
    fff_matrix_get_row(w,X,i);
    fff_vector_add(v,w);
  }
  fff_vector_scale(v,1./N);
  
  /* initilization of the precisions matrices */
  int prec_type;
  if ((int)(Precision->size1)==1)
    prec_type = 2;/*diagonal average covariance*/
  else
    if ((int)(Precision->size2)==fd2)
      prec_type = 0;/* full cluster-based covariance*/
    else {
      if ((int)(Precision->size2)==fd)
	prec_type = 1;/*diagonal cluster-based covariance*/
      else return(0);
    }
  
  switch (prec_type) {
  case 0:{   
    fff_matrix* precision = fff_matrix_new(fd, fd);
    fff_matrix* covariance = fff_matrix_new(fd, fd);
    fff_matrix_set_all( covariance,0);
    
    /* compute the global covariance */
    for (i=0 ; i<N ; i++){
      fff_matrix_get_row(w,X,i);
      fff_vector_sub(w,v);
      fff_blas_dger (1, w, w, covariance);
    }
    
    fff_matrix_scale(covariance,1.0/N);
		
	/* Derive a precision estimate */
    fff_lapack_inv_sym(precision,covariance);
	 
    for (l=0 ; l<fd ; l++)
      for (l2=0 ; l2<fd ; l2++){
	aux = fff_matrix_get(precision,l,l2); 
	for (j=0 ; j<k ; j++)
	  fff_matrix_set(Precision,j,l*fd+l2,aux); 
      }
    fff_matrix_delete(covariance);
    fff_matrix_delete(precision);
    break;
  }
  case 1:{
    fff_vector* covariance = fff_vector_new(fd);
    fff_vector* precision = fff_vector_new(fd);
    fff_vector_set_all( covariance,0);
    fff_vector_set_all( precision,1.);

    /* compute the global covariance */
    for (i=0 ; i<N ; i++){
      fff_matrix_get_row(w,X,i);
      fff_vector_sub(w,v);
      fff_vector_mul (w,w);
      fff_vector_add(covariance,w);
    }
    fff_vector_scale(covariance,1./N);
    fff_vector_div(precision,covariance);

    for (j=0 ; j<k ; j++)
      fff_matrix_set_row(Precision,j,precision);
    
    fff_vector_delete(covariance);
    fff_vector_delete(precision);
    break;
  }
  case 2:{
	
    fff_vector* covariance = fff_vector_new(fd);
    fff_vector* precision = fff_vector_new(fd);
    fff_vector_set_all( covariance,0);
    fff_vector_set_all( precision,1.);
    
    /* compute the global covariance */
    
	for (i=0 ; i<N ; i++){
      fff_matrix_get_row(w,X,i);
      fff_vector_sub(w,v);
      fff_vector_mul (w,w);
      fff_vector_add(covariance,w);
    }
    
	fff_vector_scale(covariance,1./N);
    fff_vector_div(precision,covariance);
    fff_matrix_set_row(Precision,0,precision);
	
    fff_vector_delete(covariance);
    fff_vector_delete(precision);
    
	break;
  }
  }  
    
  free(seeds); 
  fff_vector_delete(v);
  fff_vector_delete(w);
  return(1);
}

int _fff_GMM_init_hard(fff_matrix* Centers,fff_matrix* Precision, fff_vector *Weights, const fff_matrix* X, const fff_array* Label)
{
  /* char* proc = "_fff_gmm_init_hard"; */
  int fd = X->size2;   
  int fd2 = fd*fd;
  int k = Centers->size1;
  int N = X->size1;
  int i,j,l,l2;

  fff_vector *v = fff_vector_new(fd);
  fff_vector *w = fff_vector_new(fd);

  /* init the weights */
  fff_vector_set_all(Weights, 1./k);
  
  /* init the centers */
  fff_Estep(Centers,Label,X);
  
  /* init the precisions */
  int prec_type;
  if ((int)(Precision->size1)==1)
    prec_type = 2;/*diagonal average covariance*/
  else
    if ((int)(Precision->size2)==fd2)
      prec_type = 0;/* full cluster-based covariance*/
    else {
      if ((int)(Precision->size2)==fd)
	prec_type = 1;/*diagonal cluster-based covariance*/
      else return(0);
    }

  switch (prec_type) {
    case 0:{   
    fff_matrix* precision = fff_matrix_new(fd, fd);
    fff_matrix* covariance = fff_matrix_new(fd, fd);
    
    fff_matrix_set_all (covariance,0);
    double aux;

    /* compute the global covariance */
    for (i=0 ; i<N ; i++){
      fff_matrix_get_row(w,X,i);
      fff_matrix_get_row(v,Centers,fff_array_get1d(Label,i));
      fff_vector_sub(w,v);
      fff_blas_dger (1, w, w, covariance);
    }
    
    fff_matrix_scale(covariance,1.0/N);
    fff_lapack_inv_sym(precision,covariance);
    
    /* Derive a precision estimate */
    /* todo : improve this by using vector views */
    for (l=0 ; l<fd ; l++)
      for (l2=0 ; l2<fd ; l2++){
	aux = fff_matrix_get(precision,l,l2); 
	for (j=0 ; j<k ; j++)
	  fff_matrix_set(Precision,j,l*fd+l2,aux); 
      }
    fff_matrix_delete(covariance);
    fff_matrix_delete(precision);
   
    break;
  }
  case 1:{
    fff_vector* covariance = fff_vector_new(fd);
    fff_vector* precision = fff_vector_new(fd);
    fff_vector_set_all( covariance,0);
    fff_vector_set_all( precision,1.0);

    /* compute the global covariance */
    for (i=0 ; i<N ; i++){
      fff_matrix_get_row(w,X,i);
      fff_matrix_get_row(v,Centers,fff_array_get1d(Label,i));
      fff_vector_sub(w,v);
      fff_vector_mul (w,w);
      fff_vector_add(covariance,w);
    }
    fff_vector_scale(covariance,1./N);
    fff_vector_div(precision,covariance);

    for (j=0 ; j<k ; j++)
      fff_matrix_set_row(Precision,j,precision);
    
    fff_vector_delete(covariance);
    fff_vector_delete(precision);
    break;
  }
  case 2:{
    
    fff_vector* covariance = fff_vector_new(fd);
    fff_vector* precision = fff_vector_new(fd);
    fff_vector_set_all( covariance,0);
    fff_vector_set_all( precision,1.);
    
    /* compute the global covariance */
    for (i=0 ; i<N ; i++){
      fff_matrix_get_row(w,X,i);
      fff_matrix_get_row(v,Centers,fff_array_get1d(Label,i));
      fff_vector_sub(w,v);
      fff_vector_mul (w,w);
      fff_vector_add(covariance,w);
    }
    
    fff_vector_scale(covariance,1./N);
    fff_vector_div(precision,covariance);
        
    fff_matrix_set_row(Precision,0,precision);
    fff_vector_delete(covariance);
    fff_vector_delete(precision);
    
    break;
  }
  }   
  fff_vector_delete(v);
  fff_vector_delete(w);
  return(1);
}

double _fff_update_gmm(fff_matrix* Centers, fff_matrix* Precision,  fff_vector *Weights, const fff_matrix* X)
{
   char* proc = "_fff_update_gmm"; 
  double L = 0;

  int fd = X->size2;   
  int fd2 = fd*fd;
  int k = Centers->size1;
  int N = X->size1;
  int i,j,l,l1,l2;
  
  fff_matrix* Centers_new = fff_matrix_new(k, fd);
  fff_matrix* Covariance = fff_matrix_new(k,fd2);
  fff_vector* Weights_new = fff_vector_new(k);
  fff_vector* aux = fff_vector_new(fd);
  fff_vector* v = fff_vector_new(fd);
  fff_vector* w = fff_vector_new(fd);
  fff_vector* sqr_dets = fff_vector_new(k);
  fff_vector* resp = fff_vector_new(k);
  fff_matrix* precision = fff_matrix_new(fd, fd);
  fff_matrix* covariance = fff_matrix_new(fd, fd);

  fff_matrix_set_all (Centers_new,0);
  fff_vector_set_all (Weights_new,0);
  fff_matrix_set_all (Covariance,0);
  
  double temp, quad, sumr, daux;
  double thq = 4*fd;
 
  /* Pre-compute the determinants of precision matrices */
  for (j=0 ; j<k ; j++){
    for (l1=0 ; l1<fd ; l1++)
      for (l2=0 ; l2<fd ; l2++){
		temp = fff_matrix_get(Precision,j,l1*fd+l2);
		fff_matrix_set(precision,l1,l2,temp);
      }
	fff_vector_set(sqr_dets,j,sqrt(fff_lapack_det_sym(precision)));
  }

  for (i=0 ; i<N ; i++){
    fff_vector_set_all(resp,0);
    sumr = 0;
    /* compute the responsabilities */
    for (j=0 ; j<k ; j++){   
      for (l1=0 ; l1<fd ; l1++)
		for (l2=0 ; l2<fd ; l2++){
		  temp = fff_matrix_get(Precision,j,l1*fd+l2);
		  fff_matrix_set(precision,l1,l2,temp);
		}
      
      fff_matrix_get_row(aux,X,i);
      fff_matrix_get_row(v,Centers,j);
      fff_vector_sub(aux,v);
      
      fff_vector_set_all(v,0);
      fff_blas_dgemv (CblasNoTrans, 1., precision, aux, 0, v);
      quad = fff_blas_ddot (v, aux);
      
      temp = exp(-quad/2) * fff_vector_get(Weights,j) *fff_vector_get(sqr_dets,j);
      
	  fff_vector_set(resp,j,temp);
      sumr += temp;
    }
    if (sumr==0){
      sumr = exp(-thq/2);
      printf ("%s : %d %f \n",proc, i,sumr);    
    }
    
    L = L+log(sumr);

    /* update the empirical mean and covariance */
    fff_vector_scale(resp, 1./sumr);
    fff_vector_add(Weights_new,resp);
    fff_matrix_get_row(aux,X,i);
    for (j=0 ; j<k ; j++) {
      temp = fff_vector_get(resp,j);
      fff_vector_memcpy(v,aux);
      fff_vector_scale(v,temp);
      fff_matrix_get_row(w,Centers_new,j);
      fff_vector_add(v,w);
      fff_matrix_set_row(Centers_new,j,v);
      
      fff_matrix_set_all(covariance,0);
      fff_matrix_get_row(w,Centers,j);
      fff_vector_sub(w,aux);
      fff_blas_dger (1, w, w, covariance);
      for (l1=0 ; l1<fd ; l1++)
		for (l2=0 ; l2<fd ; l2++){
		  daux = fff_matrix_get(covariance,l1,l2);
		  daux *= temp;
		  fff_matrix_set(Covariance,j,l1*fd+l2,daux + fff_matrix_get(Covariance,j,l1*fd+l2)); 
		}
    }
  }
  
  /* normalize the values */ 
  for (j=0 ; j<k ; j++){
    if (fff_vector_get(Weights_new,j)==0){
      printf("%s : %d \n",proc,j);
      fff_vector_set_all(v,0);
      fff_matrix_set_row(Centers_new,j,v);
      for (l=0 ; l<fd2 ; l++)
	fff_matrix_set(Covariance,j,l,0);
    }
    else{
      temp = fff_vector_get(Weights_new,j);
      fff_matrix_get_row(w,Centers_new,j);
      fff_vector_scale(w,1./temp);
      fff_matrix_set_row(Centers_new,j,w);
      for (l=0 ; l<fd2 ; l++)
	fff_matrix_set(Covariance,j,l,fff_matrix_get(Covariance,j,l)/temp);
      temp /= N;
      fff_vector_set(Weights_new,j,temp);
    }
  }
  L /= N;
  
  /* Compute precision from the covariance */
  for (j=0 ; j<k ; j++){
    for (l1=0 ; l1<fd ; l1++)
      for (l2=0 ; l2<fd ; l2++)
	fff_matrix_set(covariance,l1,l2,fff_matrix_get(Covariance,j,l1*fd+l2));
    
    fff_lapack_inv_sym(precision,covariance);
    
    for (l1=0 ; l1<fd ; l1++)
      for (l2=0 ; l2<fd ; l2++)
	fff_matrix_set(Precision,j,l1*fd+l2,fff_matrix_get(precision,l1,l2));
  }
 
  fff_matrix_memcpy(Centers, Centers_new);
  fff_vector_memcpy(Weights, Weights_new);
 

  fff_matrix_delete(Centers_new);
  fff_matrix_delete(Covariance);
  fff_matrix_delete(covariance);
  fff_matrix_delete(precision);
  fff_vector_delete(Weights_new);
  fff_vector_delete(resp);
  fff_vector_delete(aux);
  fff_vector_delete(v);
  fff_vector_delete(w);
  fff_vector_delete(sqr_dets);
  
  return(L-0.5*fd*log(2*M_PI));
}

double _fff_update_gmm_diag(fff_matrix* Centers, fff_matrix* Precision,  fff_vector *Weights, const fff_matrix* X )
{
  char* proc = "fff_update_gmm_diag"; 
  double L = 0;

  int fd = X->size2;   
  int k = Centers->size1;
  int N = X->size1;
  int i,j,l;
  
  fff_matrix* Centers_new = fff_matrix_new(k, fd);
  fff_matrix* Covariance = fff_matrix_new(k,fd);
  fff_vector* Weights_new = fff_vector_new(k);
  fff_vector* sqr_dets = fff_vector_new(k);
  fff_vector* resp = fff_vector_new(k);
  fff_vector* v = fff_vector_new(fd);

  fff_matrix_set_all( Centers_new,0);
  fff_vector_set_all( Weights_new,0);
  fff_matrix_set_all( Covariance,0);
  
  double quad, sumr, sqr_det , aux, temp;
  double thq = 4*fd;
  
  /* Pre-compute the determinants of precision matrices */  
  for (j=0 ; j<k ; j++){
    sqr_det = 1;
    for (l=0 ; l<fd ; l++)
      sqr_det *= fff_matrix_get(Precision,j,l);
    fff_vector_set(sqr_dets,j,sqrt(sqr_det));
  }
  
  for (i=0 ; i<N ; i++){
    fff_vector_set_all( resp,0);
    sumr = 0;

    /* compute the responsabilities: quick method */
    for (j=0 ; j<k ; j++){     
      quad = 0;
      for (l=0 ; l<fd ; l++){
		aux = fff_matrix_get(X,i,l) - fff_matrix_get(Centers,j,l);
		quad += (aux * aux * fff_matrix_get(Precision,j,l));
		if (quad >thq) break;
      }
      if (quad>thq)
		fff_vector_set(resp,j,0);
      else{
		temp = fff_vector_get(sqr_dets,j)*fff_vector_get(Weights,j)*exp(-quad/2);
		fff_vector_set(resp, j, temp);
		sumr += fff_vector_get(resp,j);
      }
    }
    if (sumr==0){
      /* compute the responsabilities */
      for (j=0 ; j<k ; j++){     
		quad = 0;
		for (l=0 ; l<fd ; l++){
		  aux = fff_matrix_get(X,i,l) - fff_matrix_get(Centers,j,l);
		  temp = fff_matrix_get(Precision,j,l);
		  quad += (aux * aux * temp);
		}
		temp = fff_vector_get(sqr_dets,j)*fff_vector_get(Weights,j)*exp(-quad/2);
		fff_vector_set(resp, j, temp);
	sumr += temp;
      }
    }
    if (sumr==0){
      printf ("%s : %d %f \n", proc, i,sumr);
      sumr = exp(-thq/2);
    }
    L = L+log(sumr);
    
    /* update the empirical mean and covariance */
    fff_vector_scale(resp,1./sumr);
    for (j=0 ; j<k ; j++) {
      temp = fff_vector_get(resp,j);
      if (temp>0){
	for (l=0 ; l<fd ; l++){
	  aux = fff_matrix_get(X,i,l);
	  fff_matrix_set(Centers_new, j,l, temp*aux + fff_matrix_get(Centers_new,j,l));
	  aux -= fff_matrix_get(Centers,j,l);
	  fff_matrix_set(Covariance, j,l,aux * aux * temp+ fff_matrix_get(Covariance,j,l));
	}
      }
    }
    fff_vector_add(Weights_new,resp);
  }

  /* normalize */ 
  for (j=0 ; j<k ; j++){
    temp = fff_vector_get(Weights_new,j);
    if ( temp ==0){
      fff_vector_set_all(v,0);
      fff_matrix_set_row(Centers_new,j,v);
      fff_matrix_set_row(Covariance,j,v);
    }
    else{
      fff_matrix_get_row(v,Centers_new,j);
      fff_vector_scale(v, 1./temp);
      fff_matrix_set_row(Centers_new,j,v);
      fff_matrix_get_row(v,Covariance,j);
      fff_vector_scale(v, 1./temp);
      fff_matrix_set_row(Covariance,j,v);
    }
  }
  fff_vector_scale(Weights_new,1./N);
  L /= N;
  
  /* Compute precision from the covariance */
  for (j=0 ; j<k ; j++)
    for(l=0 ; l<fd ; l++){
      temp = fff_matrix_get(Covariance,j,l);
      if (temp>0)
	fff_matrix_set(Precision,j,l,1./temp);
      else
	fff_matrix_set(Precision,j,l,0);
    }
     
  fff_matrix_memcpy(Centers, Centers_new);
  fff_vector_memcpy(Weights, Weights_new);
   

  fff_matrix_delete(Centers_new);
  fff_matrix_delete(Covariance);
  fff_vector_delete(Weights_new);
  fff_vector_delete(resp);
  fff_vector_delete(sqr_dets);
  fff_vector_delete(v);

  return(L-0.5*fd*log(2*M_PI));
}


double _fff_update_gmm_hom(fff_matrix* Centers, fff_matrix* Precision, const fff_matrix* X)
{
  /*  char* proc = "fff_update_gmm_hom"; */
  double L = 0;

  int fd = X->size2;   
  int k = Centers->size1;
  int N = X->size1;
  int i,j,l;
  
  fff_matrix* Centers_new = fff_matrix_new(k, fd);
  fff_vector* Covariance = fff_vector_new(fd);
  fff_vector* resp = fff_vector_new(k);
  fff_vector* Weights = fff_vector_new(k);
  
  fff_vector_set_all ( Weights,0);
  fff_matrix_set_all ( Centers_new,0);
  fff_vector_set_all ( Covariance,0);
 
  double quad, sumr,sqr_det, temp, aux;
  double thq = 4*fd; 
  int auxc = 0;
  
    /* Pre-compute the determinants of precision matrices */
  sqr_det = 1;
  for (l=0 ; l<fd ; l++)
    sqr_det *= fff_matrix_get(Precision,0,l); 
  sqr_det  = sqrt(sqr_det);

  
  /* element-wise GMM update */
  
  for (i=0 ; i<N ; i++){
    fff_vector_set_all(resp,0);
    sumr = 0;
   
    /* compute the responsabilities: quick method */
    for (j=0 ; j<k ; j++){     
      quad = 0;
      for (l=0 ; l<fd ; l++){
	aux = fff_matrix_get(X,i,l) - fff_matrix_get(Centers,j,l);
	quad += (aux * aux * fff_matrix_get(Precision,0,l));
	if (quad >thq) break;
      }
      if (quad>thq)
	temp = 0;
      else
	temp = exp(-quad/2);
      fff_vector_set(resp,j,temp);
      sumr += temp;
    }
    if (sumr==0){/* compute the responsabilities: exact method */
     auxc++;
      for (j=0 ; j<k ; j++){     
	quad = 0;
	for (l=0 ; l<fd ; l++){
	  aux = fff_matrix_get(X,i,l) - fff_matrix_get(Centers,j,l);
	  quad += (aux * aux * fff_matrix_get(Precision,0,l));	  
	}
	temp = exp(-quad/2);
	sumr += temp;
	fff_vector_set(resp,j,temp);
      }
    }
    if (sumr==0){
      sumr = exp(-thq/2);
      /* fff_message (fff_NORMAL_MSG, proc, "%d %f",i,sumr); */
    }
    L = L+log(sumr);
	
    /* update the empirical mean and covariance */
    fff_vector_scale(resp,1./sumr);
    fff_vector_add(Weights,resp);
    for (j=0 ; j<k ; j++) {
      temp = fff_vector_get(resp,j);
      if (temp>0){
	  for (l=0 ; l<fd ; l++){
	  aux = fff_matrix_get(X,i,l);
	  fff_matrix_set(Centers_new, j,l, temp*aux + fff_matrix_get(Centers_new,j,l));
	  aux -= fff_matrix_get(Centers,j,l);
	  fff_vector_set(Covariance, l,aux * aux * temp+ fff_vector_get(Covariance,l));
	  }
      }
	  }
	
  }
  
  
  /* normalize */
  for (j=0 ; j<k ; j++){
    temp = fff_vector_get(Weights,j);
    if (temp==0)
      for (l=0 ; l<fd ; l++) 
	fff_matrix_set(Centers_new,j,l,0);
    else
      for (l=0 ; l<fd ; l++)
	fff_matrix_set(Centers_new,j,l,fff_matrix_get(Centers_new,j,l)/temp);
  }
  fff_vector_scale(Covariance,1./N);
 
  L /= N;
  L = L +  log(sqr_det) -  log(k);

  /* Compute precision from the covariance*/
  for (l=0 ; l<fd ; l++){
    temp = fff_vector_get(Covariance,l);
    if (temp>0)
      fff_matrix_set(Precision,0,l,1./temp);
    else
      fff_matrix_set(Precision,0,l,0);
  }
   
  fff_matrix_memcpy(Centers, Centers_new);
  
  fff_matrix_delete(Centers_new);
  fff_vector_delete(Covariance);
  fff_vector_delete(resp);
  fff_vector_delete(Weights);

  return(L-0.5*fd*log(2*M_PI));
}


double fff_gmm_mean_eval(double* L, const fff_matrix* X, const fff_matrix* Centers, const fff_matrix* Precision, const fff_vector* Weights)
{
  fff_vector * LogLike = fff_vector_new(X->size1);
  fff_array * Labels = fff_array_new1d( FFF_LONG,X->size1);
  
  fff_gmm_partition(LogLike, Labels, X, Centers, Precision,Weights);
  /* fff_gmm_eval(LogLike, X, Centers,Precision,Weights); */
  
  int i;
  *L = 0;
  for (i=0 ; i<(int)X->size1 ; i++)
    *L += fff_vector_get(LogLike,i);
  *L/=(X->size1);
  
  fff_vector_delete(LogLike);
  fff_array_delete(Labels);
  return(*L);
}


double _fff_gmm_partition(fff_array* Labels, const fff_matrix* X, const fff_matrix* Centers, const fff_matrix* Precision, const fff_vector* Weights)
{
  fff_vector * LogLike = fff_vector_new(X->size1);
  fff_gmm_partition(LogLike, Labels, X, Centers, Precision,Weights);
  double mL = 0; 
  int i;
  for (i=0 ; i<(int)X->size1 ; i++)
    mL += fff_vector_get(LogLike,i);
  mL/=(X->size1);
  fff_vector_delete(LogLike);
  return(mL);
}



int fff_gmm_partition(fff_vector* LogLike, fff_array* Labels, const fff_matrix* X, const fff_matrix* Centers, const fff_matrix* Precision, const fff_vector* Weights)
{
  if (X->size2 != Centers->size2){
    FFF_ERROR(" Inconsistant matrix sizes \n",EFAULT);
    return(0);
  }

  int fd = X->size2;   
  int fd2 = fd*fd;
  int k = Centers->size1;
  int N = X->size1;
  int i,j,l,l1,l2;
 
  double quad, sumr,sqr_det, aux, weight,hw, temp;
  double thq = 40*fd;
  double thinf = -1000;
  fff_vector *v = fff_vector_new(fd); 
 
  int prec_type;
  if ((int)(Precision->size1)==1)
    prec_type = 2;/*diagonal average covariance*/
  else
    if ((int)(Precision->size2)==fd2)
      prec_type = 0;/* full cluster-based covariance*/
    else {
      if ((int)(Precision->size2)==fd)
	prec_type = 1;/*diagonal cluster-based covariance*/
      else return(0);
    }
  
  if ((prec_type==0)&& (fd2==1))
	prec_type = 1;
  
  fff_array_set_all(Labels,-1);

  switch (prec_type) {
  case 0:{
    fff_vector * sqr_dets = fff_vector_new(k);
    fff_vector * tempx = fff_vector_new(fd);
    fff_matrix * precision = fff_matrix_new(fd,fd);

   /* Pre-compute the determinants of precision matrices */
    for (j=0 ; j<k ; j++){
      for (l1=0 ; l1<fd ; l1++)
		for (l2=0 ; l2<fd ; l2++)
		  fff_matrix_set(precision,l1,l2,fff_matrix_get(Precision,j,l1*fd+l2));
      fff_vector_set(sqr_dets,j,sqrt(fff_lapack_det_sym(precision)));
    }
	/*
	  for (j=0 ; j<k ; j++)printf("%f ",fff_vector_get(sqr_dets,j));printf("\n");
	*/
    /* element-wise likelihood */    
    for (i=0 ; i<N ; i++){
      sumr = 0;
      hw = 0;
      for (j=0 ; j<k ; j++)
		{
		  for (l1=0 ; l1<fd ; l1++)
			for (l2=0 ; l2<fd ; l2++)
			  fff_matrix_set(precision,l1,l2,fff_matrix_get(Precision,j,l1*fd+l2));
		  
		  for (l=0 ; l<fd ; l++){
			temp = fff_matrix_get(X,i,l)-fff_matrix_get(Centers,j,l);
			fff_vector_set(tempx,l,temp); 
		  }
		  fff_vector_set_all(v,0);
		  fff_blas_dgemv (CblasNoTrans, 1., precision, tempx, 0, v);
		  quad = fff_blas_ddot (v, tempx);
		  weight = exp(-quad/2) * fff_vector_get(Weights,j)*fff_vector_get(sqr_dets,j);
		  if (weight>hw){
			hw = weight;
			fff_array_set1d(Labels,i,j);
		  } 
		  sumr += weight;
		} 
      if (sumr<=0)
		fff_vector_set(LogLike,i,thinf);
	  else
		fff_vector_set(LogLike,i,log(sumr)-0.5*fd*log(2*M_PI));
    }    
    fff_vector_delete(sqr_dets);
    fff_vector_delete(tempx);
    fff_matrix_delete(precision);
    break;
  }

  case 1:{
    fff_vector * sqr_dets = fff_vector_new(k);
   /* Pre-compute the determinants of precision matrices */
    for (j=0 ; j<k ; j++){
      sqr_det = 1;
      for (l=0 ; l<fd ; l++)
	sqr_det *= (fff_matrix_get(Precision,j,l));
      fff_vector_set(sqr_dets,j,sqrt(sqr_det));
    }
    
    /* element-wise GMM update */
    for (i=0 ; i<N ; i++){
      sumr = 0;
      hw = 0;
      for (j=0 ; j<k ; j++){     
	quad = 0;
	for (l=0 ; l<fd ; l++){
	  aux = fff_matrix_get(X,i,l)-fff_matrix_get(Centers,j,l);
	  quad += (aux * aux * fff_matrix_get(Precision,j,l));
	  /**/
	  if (quad >thq) break;
	}
	if (quad<=thq){
	  weight = fff_vector_get(sqr_dets,j) * exp(-quad/2) * fff_vector_get(Weights,j);
	  if (weight>hw){
	    hw = weight;
	    fff_array_set1d(Labels,i,j);
	  } 
	  sumr += weight;
	}
      }
      if (sumr==0){  
	  for (j=0 ; j<k ; j++){     
	    quad = 0;
	    for (l=0 ; l<fd ; l++){
	      aux = fff_matrix_get(X,i,l)-fff_matrix_get(Centers,j,l);
	      quad += (aux * aux * fff_matrix_get(Precision,j,l));	  
	    } 
	    weight = fff_vector_get(sqr_dets,j) * exp(-quad/2) * fff_vector_get(Weights,j);
	    if (weight>hw){
	      hw = weight;
	      fff_array_set1d(Labels,i,j);
	    } 
	    sumr += weight;
	  }
      }
	
      if (sumr<=0)
	sumr = exp(-thq);
      fff_vector_set(LogLike,i,log(sumr) -0.5*fd*log(2*M_PI));
      
    }    
    fff_vector_delete(sqr_dets);
    break;
  }
  case 2:{ 
    /* Pre-compute the determinants of precision matrices */
    sqr_det = 1;
    for (l=0 ; l<fd ; l++)
      sqr_det *= fff_matrix_get(Precision,0,l);
    sqr_det  = sqrt(sqr_det);

    /* element-wise GMM update */
    for (i=0 ; i<N ; i++){
      sumr = 0; 
      hw = 0;
      for (j=0 ; j<k ; j++){     
	quad = 0;
	for (l=0 ; l<fd ; l++){
	  aux = fff_matrix_get(X,i,l)-fff_matrix_get(Centers,j,l);
	  quad += (aux * aux * fff_matrix_get(Precision,0,l));
	}
        weight = exp(-quad/2) * fff_vector_get(Weights,j);
	if (weight>hw){
	  hw = weight;
	  fff_array_set1d(Labels,i,j);
	} 
	sumr += weight;
      }
      if (sumr<=0)
	sumr = exp(-thq);
      fff_vector_set(LogLike,i,log(sumr) -0.5*fd*log(2*M_PI) +  log(sqr_det) );
    }
    break;
  }}
  fff_vector_delete(v);
  return(1);
}


extern int fff_gmm_membership(fff_graph* G, const fff_matrix* X, const fff_matrix* Centers, const fff_matrix* Precision, const fff_vector* Weights)
{
  if (X->size2 != Centers->size2)
    FFF_ERROR("Inconsistant matrix sizes \n",EFAULT);

  int fd = X->size2;
  int fd2 = fd*fd;
  int k = Centers->size1;
  int N = X->size1;
  int i,j,l,l1,l2;
  int edges = 0;

  double quad, sumr,sqr_det, aux, weight,temp;
  double  thq = 4*fd;
  fff_vector *v = fff_vector_new(fd); 
   fff_vector *TotalLike = fff_vector_new(N);
  

  int prec_type;
  if ((int)(Precision->size1)==1)
    prec_type = 2;/*diagonal average covariance*/
  else
    if ((int)(Precision->size2)==fd2)
      prec_type = 0;/* full cluster-based covariance*/
    else {
      if ((int)(Precision->size2)==fd)
	prec_type = 1;/*diagonal cluster-based covariance*/
      else return(0);
    }

  switch (prec_type) {
  case 0:{
    fff_vector * sqr_dets = fff_vector_new(k);
    fff_vector * tempx = fff_vector_new(fd);
    fff_matrix * precision = fff_matrix_new(fd,fd);

   /* Pre-compute the determinants of precision matrices */
    for (j=0 ; j<k ; j++){
      for (l1=0 ; l1<fd ; l1++)
	for (l2=0 ; l2<fd ; l2++)
	  fff_matrix_set(precision,l1,l2,fff_matrix_get(Precision,j,l1*fd+l2));
      fff_vector_set(sqr_dets,j,sqrt(fff_lapack_det_sym(precision)));
    }

    /* element-wise likelihood */ 
    for (i=0 ; i<N ; i++){
      sumr = 0;
      for (j=0 ; j<k ; j++)
		{
		  for (l1=0 ; l1<fd ; l1++)
			for (l2=0 ; l2<fd ; l2++)
			  fff_matrix_set(precision,l1,l2,fff_matrix_get(Precision,j,l1*fd+l2));
		  
		  for (l=0 ; l<fd ; l++){
			temp = fff_matrix_get(X,i,l)-fff_matrix_get(Centers,j,l);
			fff_vector_set(tempx,l,temp); 
		  }
		  fff_vector_set_all(v,0);
		  fff_blas_dgemv (CblasNoTrans, 1., precision, tempx, 0, v);
		  quad = fff_blas_ddot (v, tempx);
		  if (quad>thq){
			weight = exp(-quad/2) * fff_vector_get(Weights,j)*fff_vector_get(sqr_dets,j);
			if (G->E>0){
			  G->eA[edges] = i;
			  G->eB[edges] = j;
			  G->eD[edges] = weight;
			}
			edges++; 
			sumr += weight;
		  }
		} 
      if (sumr<=0) sumr = exp(-thq/2);
	  /* retain this for normalization */
      fff_vector_set(TotalLike,i,sumr);
    }    
    fff_vector_delete(sqr_dets);
    fff_vector_delete(tempx);
    fff_matrix_delete(precision);
    break;
  }

  case 1:{
    fff_vector * sqr_dets = fff_vector_new(k);
   /* Pre-compute the determinants of precision matrices */
    for (j=0 ; j<k ; j++){
      sqr_det = 1;
      for (l=0 ; l<fd ; l++)
	sqr_det *= (fff_matrix_get(Precision,j,l));
      fff_vector_set(sqr_dets,j,sqrt(sqr_det));
    }
    
    /* element-wise GMM update */
    for (i=0 ; i<N ; i++){
      sumr = 0;
	  for (j=0 ; j<k ; j++){     
		quad = 0;
		for (l=0 ; l<fd ; l++){
		  aux = fff_matrix_get(X,i,l)-fff_matrix_get(Centers,j,l);
		  quad += (aux * aux * fff_matrix_get(Precision,j,l));
		  if (quad >thq) break;
		}
		if (quad<=thq){
		  weight = fff_vector_get(sqr_dets,j) * exp(-quad/2) * fff_vector_get(Weights,j);
		  if (G->E>0){
			G->eA[edges] = i;
			G->eB[edges] = j;
			G->eD[edges] = weight;
		  }
		  sumr += weight;
		  edges++;
		}
      }
      if (sumr==0){  
	/* printf("%f %d ",sumr,edges);*/
		for (j=0 ; j<k ; j++){     
		  quad = 0;
		  for (l=0 ; l<fd ; l++){
			aux = fff_matrix_get(X,i,l)-fff_matrix_get(Centers,j,l);
			quad += (aux * aux * fff_matrix_get(Precision,j,l));	  
		  } 
		  weight = fff_vector_get(sqr_dets,j) * exp(-quad/2) * fff_vector_get(Weights,j);
		  if (G->E>0){	
			G->eA[edges] = i;
			G->eB[edges] = j;
			G->eD[edges] = weight;
		  }
		  sumr += weight;
		  edges++;
		}
      }
	  
      if (sumr<=0) sumr = exp(-thq);
      fff_vector_set(TotalLike,i,sumr);
      /* printf("%d %d ",i,edges); */
    }    
    fff_vector_delete(sqr_dets);
    break;
  }
  case 2:{ 
    /* Pre-compute the determinants of precision matrices */
    sqr_det = 1;
    for (l=0 ; l<fd ; l++)
      sqr_det *= fff_matrix_get(Precision,0,l);
    sqr_det  = sqrt(sqr_det);

    /* element-wise GMM update */
    for (i=0 ; i<N ; i++){
      sumr = 0; 
	  for (j=0 ; j<k ; j++){     
		quad = 0;
		for (l=0 ; l<fd ; l++){
		  aux = fff_matrix_get(X,i,l)-fff_matrix_get(Centers,j,l);
		  quad += (aux * aux * fff_matrix_get(Precision,0,l));
		}
		if (quad>thq){
		   weight = exp(-quad/2) * fff_vector_get(Weights,j);
		   if (G->E>0){
			  G->eA[edges] = i;
			  G->eB[edges] = j;
			  G->eD[edges] = weight;
			}
			edges++;
			sumr += weight;
		 }
      }
      if (sumr<=0) sumr = exp(-thq);
      fff_vector_set(TotalLike,i,sumr);
    }
    break;
  }}
  
  /* renormalize the probabilities */
  for (i=0;i<G->E;i++) 
	G->eD[i] = 	G->eD[i]/fff_vector_get(TotalLike,G->eA[i]);

  fff_vector_delete(v);
  fff_vector_delete(TotalLike);
  return(edges);
}

extern int fff_gmm_shift(fff_matrix* X, const fff_matrix* Centers, const fff_matrix* Precision, const fff_vector* Weights)
{
  if (X->size2 != Centers->size2)
    FFF_ERROR(" Inconsistent matrix sizes ",EFAULT);
 

  int fd = X->size2;
  int fd2 = fd*fd;
  int k = Centers->size1;
  int N = X->size1;
  int i,j,l;

  double quad, sumr,sqr_det, aux, weight;
  double  thq = 4*fd;
  fff_vector *v =  fff_vector_new(fd);
  fff_vector *w =  fff_vector_new(fd);

  int prec_type;
  if ((int)(Precision->size1)==1)
    prec_type = 2;/*diagonal average covariance*/
  else
    if ((int)(Precision->size2)==fd2)
      prec_type = 0;/* full cluster-based covariance*/
    else {
      if ((int)(Precision->size2)==fd)
	prec_type = 1;/*diagonal cluster-based covariance*/
      else return(0);
    }

  switch (prec_type) {
  case 0:{
	printf("Not implemented yet; use the diagonal precision model instead. \n");
	return(0);
  }

  case 1:{
    fff_vector * sqr_dets = fff_vector_new(k);
   /* Pre-compute the determinants of precision matrices */
    for (j=0 ; j<k ; j++){
      sqr_det = 1;
      for (l=0 ; l<fd ; l++)
	sqr_det *= (fff_matrix_get(Precision,j,l));
      fff_vector_set(sqr_dets,j,sqrt(sqr_det));
    }
    
    /* element-wise GMM update */
    for (i=0 ; i<N ; i++){
      sumr = 0;
	  fff_vector_set_all(v,0);
	  fff_vector_set_all(w,0);
	  for (j=0 ; j<k ; j++){     
		quad = 0;
		for (l=0 ; l<fd ; l++){
		  aux = fff_matrix_get(X,i,l)-fff_matrix_get(Centers,j,l);
		  quad += (aux * aux * fff_matrix_get(Precision,j,l));
		  if (quad >thq) break;
		}
		if (quad<=thq){
		  weight = fff_vector_get(sqr_dets,j) * exp(-quad/2) * fff_vector_get(Weights,j);
		  
		  for (l=0 ; l<fd ; l++){
			aux = fff_vector_get(v,l);
			aux += weight* fff_matrix_get(Precision,j,l)*fff_matrix_get(Centers,j,l);
			fff_vector_set(v,l,aux);
			aux = fff_vector_get(w,l);
			aux += weight* fff_matrix_get(Precision,j,l);
			fff_vector_set(w,l,aux);
		  }
		  sumr += weight;
		}
      }
      if (sumr==0){  
		for (j=0 ; j<k ; j++){     
		  quad = 0;
		  for (l=0 ; l<fd ; l++){
			aux = fff_matrix_get(X,i,l)-fff_matrix_get(Centers,j,l);
			quad += (aux * aux * fff_matrix_get(Precision,j,l));	  
		  } 
		  weight = fff_vector_get(sqr_dets,j) * exp(-quad/2) * fff_vector_get(Weights,j);
		  for (l=0 ; l<fd ; l++){
			aux = fff_vector_get(v,l);
			aux += weight* fff_matrix_get(Precision,j,l)*fff_matrix_get(Centers,j,l);
			fff_vector_set(v,l,aux);
			aux = fff_vector_get(w,l);
			aux += weight* fff_matrix_get(Precision,j,l);
			fff_vector_set(w,l,aux);
		  }
		  sumr += weight;
		}
      }
	  /* perform the shift now */
	  if (sumr>0)
		for (l=0 ; l<fd ; l++){
		  aux = fff_vector_get(v,l)/fff_vector_get(w,l);
		  fff_matrix_set(X,i,l,aux);
		}    
	}
	fff_vector_delete(sqr_dets);
	break;
  }
	case 2:{ 
	  /* Pre-compute the determinants of precision matrices */
	  sqr_det = 1;
	  for (l=0 ; l<fd ; l++)
		sqr_det *= fff_matrix_get(Precision,0,l);
	  sqr_det  = sqrt(sqr_det);
	  
	  /* element-wise GMM update */
	  for (i=0 ; i<N ; i++){
		sumr = 0;  
		fff_vector_set_all(v,0);
		fff_vector_set_all(w,0);
		for (j=0 ; j<k ; j++){     
		  quad = 0;
		  for (l=0 ; l<fd ; l++){
			aux = fff_matrix_get(X,i,l)-fff_matrix_get(Centers,j,l);
			quad += (aux * aux * fff_matrix_get(Precision,0,l));
		  }
		  
		  if (quad>thq){
			weight = exp(-quad/2) * fff_vector_get(Weights,j);
			for (l=0 ; l<fd ; l++){
			  aux = fff_vector_get(v,l);
			  aux += weight* fff_matrix_get(Precision,j,l)*fff_matrix_get(Centers,j,l);
			  fff_vector_set(v,l,aux);
			  aux = fff_vector_get(w,l);
			  aux += weight* fff_matrix_get(Precision,j,l);
			  fff_vector_set(w,l,aux);
			}
			sumr += weight;
		  }
		}
		/* perform the shift now */
		if (sumr>0)
		  for (l=0 ; l<fd ; l++){
			aux = fff_vector_get(v,l)/fff_vector_get(w,l);
		  fff_matrix_set(X,i,l,aux);
		  }    
	  }
	  break;
	}}
	
  fff_vector_delete(v);
  fff_vector_delete(w);
  return(0);
}

/**********************************************************************
 ********************* Variational Bayesian GMM ***********************
**********************************************************************/


extern double fff_VB_gmm_estimate(fff_Bayesian_GMM* BGMM,  fff_array *Label, const fff_matrix* X, const int maxiter, const double delta)
{
  char* proc = "fff_VB_gmm_estimate";
  int i;
  fff_vector* Like = fff_vector_new(maxiter);
  double L0 = 0;
  double La = 0;
  int verbose = 0;
 
  _fff_VBGMM_init(BGMM);
 
  if (fff_clustering_OntoLabel(Label,BGMM->k)){
	_fff_GMM_init_hard(BGMM->means,BGMM->precisions,BGMM->weights,X,Label);
  }
  double aux;
  int j,l;
  for (j=0 ; j<BGMM->k ; j++)  for (l=0 ; l<BGMM->dim ; l++){
	  aux = fff_matrix_get(BGMM->precisions,j,l)/fff_vector_get(BGMM->dof,j); 
	  fff_matrix_set(BGMM->precisions,j,l,aux);
  }

  for (i=0; i<maxiter; i++){
    Like->data[i] = _fff_VB_update_gmm(X,BGMM);  
    if (verbose)
	  printf ("%s : it %d LL=%f\n",proc, i,Like->data[i]);
    if (i>1){
      if (delta*(La-L0)>Like->data[i]-La){
	La = Like->data[i];
	break; 
      }
    }
    else
      L0 = Like->data[i];
    La = Like->data[i];
  }
  
  /* take the map labelling */
  _fff_VB_gmm_MAP(Label,X,BGMM);
  /*La = _fff_gmm_partition(Label, X, BGMM->means, BGMM->precisions, BGMM->weights);*/
  
  fff_vector_delete( Like );


  return(La);
}

extern int fff_BGMM_sampling(fff_vector* density, fff_Bayesian_GMM* BG, const fff_matrix *grid)
{
  double LL;
  fff_vector * proba = fff_vector_new(BG->k);
  int i;
  fff_vector *x = fff_vector_new(BG->dim);

  for (i=0 ; i<grid->size1 ; i++){
	fff_matrix_get_row(x,grid,i); 
	LL = _fff_WNpval_(proba,x,BG);
	fff_vector_set(density,i,LL);
  }

  fff_vector_delete(x);
  fff_vector_delete(proba);
  return 0;
}

/*
extern int _fff_BGMM_sampling(fff_matrix* membership, fff_Bayesian_GMM* BG, const fff_matrix *grid)
{

  fff_WNpval(membership, grid,BG);
  
  fff_vector *x = fff_vector_new(BG->dim);
  double LL;
  fff_vector * proba = fff_vector_new(BG->k);
  int i;
  for (i=0 ; i<grid->size1 ; i++){
	fff_matrix_get_row(x,grid,i); 
	_fff_WNpval_(proba,x,BG);
	fff_matrix_set_row (memebsership,i, proba);
  }

  fff_vector_delete(x);
  fff_vector_delete(proba);
  return 0;
  
}
*/

static int _fff_VBGMM_init(fff_Bayesian_GMM* BGMM)
{
  
  fff_vector_memcpy(BGMM->means_scale,BGMM->prior_means_scale);
  fff_vector_memcpy(BGMM->dof,BGMM->prior_dof);
  fff_matrix_memcpy (BGMM->means, BGMM->prior_means);
  fff_matrix_memcpy (BGMM->precisions, BGMM->prior_precisions);
  double k = BGMM->k;
  fff_vector_set_all(BGMM->weights,1.0/k);

  return(0);
}


int _fff_VB_gmm_MAP(fff_array *Label, const fff_matrix* X, const fff_Bayesian_GMM* BGMM)
{
  /* this function performs the MAP assignment of the elements of X */ 

  int fd = BGMM->dim;   
  int k = BGMM->k;
  int N = X->size1;
  int i,j,l;
  
  fff_vector* log_norm_fact = fff_vector_new(k);
  
  double quad, aux, dof, thq;

  
  /* Pre-compute the normalizing factors */
  /* analogous to the determinants of the precisions */
  /* and the weights */ 
  _fff_VB_log_norm(log_norm_fact, BGMM);

  for (i=0 ; i<N ; i++){
	thq = FFF_NEGINF;

    for (j=0 ; j<k ; j++){     
      quad = 2*fff_vector_get(log_norm_fact,j);
	  dof = fff_vector_get(BGMM->dof,j);
      for (l=0 ; l<fd ; l++){
		aux = fff_matrix_get(X,i,l) - fff_matrix_get(BGMM->means,j,l);
		quad -= (dof * aux * aux * fff_matrix_get(BGMM->precisions,j,l));
		/*if (quad <thq) break;*/
      }
	  
      if (quad>thq){
		thq = quad;
		fff_array_set1d(Label, i, j);
      }
    }
  }
  return 0;
}

int _fff_VB_log_norm(fff_vector* log_norm_fact, const fff_Bayesian_GMM* BGMM)
{
  int j,l,k = BGMM->k;
  int fd =  BGMM->dim;
  double sqr_det,temp;
  
   for (j=0 ; j<k ; j++){
    sqr_det = -(double)(fd)/fff_vector_get(BGMM->means_scale,j);
	sqr_det += (double)(fd)*log(2);
    for (l=0 ; l<fd ; l++){
      temp = fff_vector_get(BGMM->dof,j)-l;
      if (temp>0)
		sqr_det += fff_psi(temp/2);
   
      sqr_det += log(fff_matrix_get(BGMM->precisions,j,l));  
 	}    
	sqr_det/=2;
	sqr_det += log(fff_vector_get(BGMM->weights,j));
	fff_vector_set(log_norm_fact,j,sqr_det);
   }
   return 0;
}

double _fff_VB_update_gmm(const fff_matrix* X, fff_Bayesian_GMM* BGMM)
{
  /* this function performs one complete VB-EM iteration of the BGMM */ 
  char* proc = "fff_VB_update_gmm"; 
  double L = 0;

  int fd = BGMM->dim;   
  int k = BGMM->k;
  int N = X->size1;
  int i,j,l;
  
  fff_matrix* emp_Centers = fff_matrix_new(k, fd);
  fff_matrix* Covariance = fff_matrix_new(k,fd);
  fff_vector* emp_Weights = fff_vector_new(k);
  fff_vector* log_norm_fact = fff_vector_new(k);
  fff_vector* resp = fff_vector_new(k);
  fff_vector* v = fff_vector_new(fd);
  fff_vector* w = fff_vector_new(fd);
  fff_vector* x = fff_vector_new(fd);
  fff_vector* y = fff_vector_new(fd);

  fff_matrix_set_all( emp_Centers,0);
  fff_vector_set_all( emp_Weights,0);
  fff_matrix_set_all( Covariance,0);
  
  double quad, sumr, aux, temp,wl,dof;
  double thq = 4*fd;
  
  /* Pre-compute the normalizing factors */
  /* analogous to the determinants of the precisions */
  /* and the weights */
  _fff_VB_log_norm(log_norm_fact, BGMM);

  for (j=0 ; j<k ; j++) {
	if ((fff_vector_get(log_norm_fact,j))==FFF_NAN) return 0;
  }

  for (i=0 ; i<N ; i++){
    fff_vector_set_all( resp,0);
    sumr = 0;
	
    /* compute the responsabilities: quick and approximative method */
    for (j=0 ; j<k ; j++){     
      quad = 0;
	  dof = fff_vector_get(BGMM->dof,j);
      for (l=0 ; l<fd ; l++){
		aux = fff_matrix_get(X,i,l) - fff_matrix_get(BGMM->means,j,l);
		temp = fff_matrix_get(BGMM->precisions,j,l);
		quad += (dof * aux * aux * temp);
		if (quad >thq) break;
      }
      if (quad>thq)
		fff_vector_set(resp,j,0);
      else{
		temp = fff_vector_get(log_norm_fact,j)-quad/2;
		temp = exp(temp);
		fff_vector_set(resp, j, temp);
		sumr += temp;
      }
    }
    if (sumr==0){
	  /* restart with the slow method */
      /* compute the responsabilities */
      for (j=0 ; j<k ; j++){     
		quad = 0;
		dof = fff_vector_get(BGMM->dof,j);
		for (l=0 ; l<fd ; l++){
		  aux = fff_matrix_get(X,i,l) - fff_matrix_get(BGMM->means,j,l);
		  temp = fff_matrix_get(BGMM->precisions,j,l);
		  quad += (dof * aux * aux * temp);
		}
		temp = fff_vector_get(log_norm_fact,j)-quad/2;
		temp = exp(temp);
		fff_vector_set(resp, j, temp);
		sumr += temp;
      }
    }
    if (sumr==0){
	  printf ("%s : %d %f \n",proc, i,sumr);
      sumr = exp(-thq/2);
    }
    L = L+log(sumr);
    
    /* update the empirical mean and covariance */
    fff_vector_scale(resp,1./sumr);
    for (j=0 ; j<k ; j++) {
      wl = fff_vector_get(resp,j);
      if (wl>0){
		for (l=0 ; l<fd ; l++){
		  aux = fff_matrix_get(X,i,l);
		  temp = wl*aux + fff_matrix_get(emp_Centers,j,l);
		  fff_matrix_set(emp_Centers, j,l, temp);

		  /*beware:this should be emp_Centers*/
		  aux -= fff_matrix_get(BGMM->means,j,l); 
		  temp = aux * aux * wl+ fff_matrix_get(Covariance,j,l);
		  fff_matrix_set(Covariance, j,l,temp);
		}
      }
    }
    fff_vector_add(emp_Weights,resp);
  }

  /* Compute the posteriorof mean, meas scale, and dof */ 
  
  /* the scale of the mean */
  fff_vector_memcpy(BGMM->means_scale,emp_Weights);
  fff_vector_add(BGMM->means_scale,BGMM->prior_means_scale);
  /* dof */
  fff_vector_memcpy(BGMM->dof,emp_Weights);
  fff_vector_add(BGMM->dof,BGMM->prior_dof);

  for (j=0 ; j<k ; j++){
    temp = fff_vector_get(emp_Weights,j);	

	/* mean */
    fff_matrix_get_row(v,BGMM->prior_means,j);
    fff_vector_scale(v,fff_vector_get(BGMM->prior_means_scale,j));
    fff_matrix_get_row(w,emp_Centers,j);
    fff_vector_add(v,w);
	fff_vector_scale(v, 1./fff_vector_get(BGMM->means_scale,j));
    fff_matrix_set_row(BGMM->means,j,v);
  }

  
  /* take the posterior variance */ 
  double eps = (1./N)*1.e-4;
  for (j=0 ; j<k ; j++){
	
	for(l=0 ; l<fd ; l++){
	  aux = fff_matrix_get(BGMM->prior_precisions,j,l);
	  fff_vector_set(w,l,1./aux);
	}

    fff_matrix_get_row(v,Covariance,j);
    fff_vector_add(v,w);

    temp = fff_vector_get(emp_Weights,j);
    if (temp>eps){
      fff_matrix_get_row(x,emp_Centers,j);
      fff_vector_scale(x,1./temp);
    }
    else{
      fff_matrix_get_row(x,BGMM->means,j);
	}

    fff_matrix_get_row(y,BGMM->prior_means,j);
    fff_vector_sub(x,y);
    fff_vector_mul(x,x);
    fff_vector_scale(x, temp);
	fff_vector_scale(x,fff_vector_get(BGMM->prior_means_scale,j));	
	fff_vector_scale(x,1./fff_vector_get(BGMM->means_scale,j));
    fff_vector_add(v,x);
   
    fff_matrix_set_row(Covariance,j,v);
  }
  
  /* Compute precision from the covariance */
  for (j=0 ; j<k ; j++)
    for(l=0 ; l<fd ; l++){
      temp = fff_matrix_get(Covariance,j,l);
	  fff_matrix_set(BGMM->precisions,j,l,1./temp);
	}
     

  /* Finally, update the weights */
  fff_vector *lambda = fff_vector_new(k);
  double slambda = 0;
  for (j=0 ; j<k ; j++){
    temp = fff_vector_get(emp_Weights,j)+fff_vector_get(BGMM->prior_weights,j);
    slambda += temp;
    fff_vector_set(lambda,j,fff_psi(temp));
  }
  fff_vector_add_constant (lambda, -fff_psi(slambda));
  for (j=0 ; j<k ; j++)
    fff_vector_set(BGMM->weights,j, exp(fff_vector_get(lambda,j)));
  fff_vector_delete(lambda);
 

  fff_matrix_delete(emp_Centers);
  fff_matrix_delete(Covariance);
  fff_vector_delete(emp_Weights);
  fff_vector_delete(resp);
  fff_vector_delete(log_norm_fact);
  fff_vector_delete(v);
  fff_vector_delete(w); 
  fff_vector_delete(x);
  fff_vector_delete(y);

  L /= N; /*approximate*/
  return(L-0.5*fd*log(2*M_PI));

}

/************************************************************************/
/********************** Gibbs sampling for GMM  *************************/
/************************************************************************/


/* "contructor" */
extern fff_Bayesian_GMM* fff_BGMM_new( const long k, const long dim )
{
  fff_Bayesian_GMM* thisone;  

  /* Start with allocating the object */
  thisone = (fff_Bayesian_GMM*) calloc( 1, sizeof(fff_Bayesian_GMM) );
  
  /* Checks that the pointer has been allocated */
  if ( thisone == NULL) 
    return NULL; 

  /* Initialization */
  thisone->k = k;
  thisone->dim = dim;

  /* Allocate BGMM objects */

  thisone->prior_means = fff_matrix_new(k,dim);
  thisone->prior_means_scale = fff_vector_new(k);
  thisone->prior_precisions = fff_matrix_new(k,dim);
  thisone->prior_dof = fff_vector_new(k);
  thisone->prior_weights = fff_vector_new(k);
  
  thisone->means = fff_matrix_new(k,dim);
  thisone->means_scale = fff_vector_new(k);
  thisone->precisions = fff_matrix_new(k,dim);
  thisone->dof = fff_vector_new(k);
  thisone->weights = fff_vector_new(k);

  /* Allocation test */ 
  int b = (thisone->prior_means ==NULL);
  /* TODO...*/
  /* b = b | (); */

  if (b) {
    fff_BGMM_delete( thisone );
    return NULL;
  }

  return thisone;  
}

/* "destructor" */
extern int fff_BGMM_delete( fff_Bayesian_GMM* thisone )
{
  if ( thisone != NULL ) {
	fff_matrix_delete(thisone->prior_means);
	fff_vector_delete(thisone->prior_means_scale);
	fff_matrix_delete(thisone->prior_precisions);
	fff_vector_delete(thisone->prior_dof);
	fff_vector_delete(thisone->prior_weights);

	fff_matrix_delete(thisone->means);
	fff_vector_delete(thisone->means_scale);
	fff_matrix_delete(thisone->precisions);
	fff_vector_delete(thisone->dof);
	fff_vector_delete(thisone->weights);

	free(thisone);
  }
  return(0);
}

/* instantiate with priors */
extern int fff_BGMM_set_priors(fff_Bayesian_GMM* BG, const fff_matrix * prior_means, const fff_vector *prior_means_scale, const fff_matrix * prior_precisions, const fff_vector* prior_dof, const fff_vector *prior_weights )
{
  /* check the dimensions ...*/
  fff_matrix_memcpy(BG->prior_means,prior_means );
  fff_vector_memcpy(BG->prior_means_scale,prior_means_scale );
  fff_matrix_memcpy(BG->prior_precisions,prior_precisions );
  fff_vector_memcpy(BG->prior_dof,prior_dof);
  fff_vector_memcpy(BG->prior_weights,prior_weights);
  return(0);
}

static int _fff_BGMM_init(fff_Bayesian_GMM* BG)
{
  int i,j;
  double a,b,ms;

  fff_vector_memcpy(BG->means_scale,BG->prior_means_scale );
  fff_vector_memcpy(BG->weights,BG->prior_weights );
  fff_vector_memcpy(BG->dof,BG->prior_dof);
  fff_matrix_memcpy(BG->precisions,BG->prior_precisions );

  fff_matrix * mprecision = fff_matrix_new(BG->k,BG->dim);

  for (i=0 ; i<BG->k ; i++){
	a = fff_vector_get(BG->dof,i);
	ms = fff_vector_get(BG->means_scale,i);
	for (j=0 ; j<BG->dim ; j++){
	  b = fff_matrix_get(BG->precisions,i,j);
	  b*=a;
	  fff_matrix_set(mprecision,i,j,b*ms);
	}
  }
  
  generate_normals(BG->means, BG->prior_means, mprecision);
   
  fff_matrix_delete(mprecision);
  return(0);
}

static double _fff_WNpval_(fff_vector * proba, const fff_vector *X, const fff_Bayesian_GMM* BG)
{
   int i,j;
   double x,tau,m,w,a,b,f,sxw,LL=0;
  
   sxw = 0;
   for (i=0 ; i<BG->k ; i++){
	 w = 0;
	 f = 0;
	 a = fff_vector_get(BG->dof,i);
	 tau = fff_vector_get(BG->means_scale,i);
	 tau = tau/(1+tau);
	 for (j=0 ; j<BG->dim ; j++){
	   m = fff_matrix_get(BG->means,i,j);
	   b = fff_matrix_get(BG->precisions,i,j);
	   x = fff_vector_get(X,j);
	   f = f+ log( 1./b + tau* (m-x)*(m-x));
	   w = w - log(b)*a; 
	   w += 2*fff_gamln((a+1-j)/2);
	   w -= 2*fff_gamln((a-j)/2);
	 }
	 w = w -f*(a+1) + log(tau)*BG->dim;
	 w = w-log(M_PI)*BG->dim;
	 w = w/2;
	 w = w + log(fff_vector_get(BG->weights,i));
	 w = exp(w);
	 sxw += w;
	 fff_vector_set(proba,i,w);
   }
   LL += log(sxw);
   return(LL);
}

extern double fff_WNpval(fff_matrix * proba, const fff_matrix *X, const fff_Bayesian_GMM* BG)
{
   int i,j,n;
   double x,tau,m,w,a,b,f,sxw,LL=0;
  
   for (n=0 ; n<X->size1 ; n++){
	 sxw = 0;
	 for (i=0 ; i<BG->k ; i++){
	   w = 0;
	   f = 0;
	   a = fff_vector_get(BG->dof,i);
	   tau = fff_vector_get(BG->means_scale,i);
	   tau = tau/(1+tau);
	   for (j=0 ; j<BG->dim ; j++){
		 m = fff_matrix_get(BG->means,i,j);
		 b = fff_matrix_get(BG->precisions,i,j);
		 x = fff_matrix_get(X,n,j);
		 f = f+ log( 1./b + tau* (m-x)*(m-x));
		 w = w - log(b)*a; 
		 w += 2*fff_gamln((a+1-j)/2);
		 w -= 2*fff_gamln((a-j)/2);
	   }
	   w = w -f*(a+1) + log(tau)*BG->dim;
	   w = w-log(M_PI)*BG->dim;
	   w = w/2;
	   w = w + log(fff_vector_get(BG->weights,i));
	   w = exp(w);
	   sxw += w;
	   fff_matrix_set(proba,n,i,w);
	 }
	 LL += log(sxw);
   }
   return(LL/X->size1);
}

static double _fff_Npval(fff_matrix * proba, const fff_matrix *X, const fff_Bayesian_GMM* BG)
{
  int i,j,n;
  double tau,m,w,p,x,sxw,a;
  double LL=0;

  for (n=0 ; n<X->size1 ; n++){
	sxw = 0;
	for (i=0 ; i<BG->k ; i++){
	  w = 0;
	  tau = fff_vector_get(BG->means_scale,i);
	  tau = tau/(1+tau);
	  a = fff_vector_get(BG->dof,i);
	  for (j=0 ; j<BG->dim ; j++){
		m = fff_matrix_get(BG->means,i,j);
		p = fff_matrix_get(BG->precisions,i,j)*a;
		x = fff_matrix_get(X,n,j);
		w = w + log(tau) + log(p) - tau*(m-x)*(m-x)*p; 
	  }
	  w = w-log(2*M_PI)*BG->dim;
	  w = w/2;
	  w = w + log(fff_vector_get(BG->weights,i));
	  w = exp(w);
	  fff_matrix_set(proba,n,i,w);
	  sxw += w;
	} 
	LL += log(sxw);
  }
  return(LL/X->size1);
}


static int _fff_random_choice(fff_array *choice, fff_vector * pop, const fff_matrix * proba, int nit)
{
  int n,j;
  rk_state state; 
  rk_seed(nit, &state);
  
  double sp,h,cou;
  
  
  for (n=0 ; n<proba->size1 ; n++){
	sp = 0;
	for (j=0 ; j<proba->size2 ; j++)
	  sp += fff_matrix_get(proba,n,j);
	
	h = rk_double(&state)*sp;
	sp = 0;
	for (j=0 ; j<proba->size2 ; j++){
	  sp +=fff_matrix_get(proba,n,j);
	  if (sp>=h) break;
	}
  
	fff_array_set1d(choice,n,j);
	cou = fff_vector_get(pop,j);
	fff_vector_set(pop,j,cou+1);
	
  }
  
  return 0;
}

static double _fff_update_BGMM(fff_Bayesian_GMM* BG, const fff_matrix *X, int nit, const int method)
{
  double LL=0;
  fff_matrix * proba = fff_matrix_new(X->size1,BG->k);
  LL = _fff_full_update_BGMM(proba, BG, X, nit, method);
  fff_matrix_delete(proba);
  return LL;
}

static double _fff_full_update_BGMM(fff_matrix * proba, fff_Bayesian_GMM* BG, const fff_matrix *X, int nit, const int method)
{
  int i,j,n;
  double sw,x,a,b,dx,LL=0;

  fff_vector * pop = fff_vector_new(BG->k);
  fff_matrix * means = fff_matrix_new (BG->k,BG->dim);
  fff_matrix * variance = fff_matrix_new (BG->k,BG->dim);
  fff_array *choice = fff_array_new1d(FFF_LONG,X->size1);
  
  
  if (method == 0) _fff_Npval(proba,X,BG);
  else LL = fff_WNpval(proba,X,BG);
  _fff_random_choice(choice,pop,proba,nit);
  
  /* update the weight*/
  fff_vector_memcpy(BG->weights,BG->prior_weights );
  fff_vector_add(BG->weights,pop);
  sw=0;
  for (i=0 ; i<BG->k ; i++) 
	sw += fff_vector_get(BG->weights,i);
  fff_vector_scale(BG->weights,1./sw);
  
  /*compute component_wise sums*/
  for (n=0 ; n<X->size1 ; n++){
	i = fff_array_get1d(choice,n);
	for (j=0 ; j<BG->dim ; j++){
	  x = fff_matrix_get(X,n,j) + fff_matrix_get(means,i,j);
	  fff_matrix_set(means,i,j,x);
	}
  }

  /* update the means*/
  fff_vector_memcpy(BG->means_scale,BG->prior_means_scale);
  fff_vector_add(BG->means_scale,pop);
  
  for( i=0 ; i<BG->k ; i++ ){
	a = fff_vector_get(BG->prior_means_scale,i);
	b = fff_vector_get(BG->means_scale,i);
	for (j=0 ; j<BG->dim ; j++){
	  x = fff_matrix_get(BG->prior_means,i,j)*a + fff_matrix_get(means,i,j);
	  x /= b;
	  fff_matrix_set(BG->means,i,j,x);
	}
  }
    
  /* compute the variance */
  for (n=0 ; n<X->size1 ; n++){
	
	i = fff_array_get1d(choice,n);
	for (j=0 ; j<BG->dim ; j++){
	  x =  fff_matrix_get(variance,i,j);
	  dx = fff_matrix_get(X,n,j) - fff_matrix_get(BG->means,i,j);
	  fff_matrix_set(variance,i,j,x+dx*dx);
	}
  }

  /* update the precision */
  fff_vector_memcpy(BG->dof,BG->prior_dof);
  fff_vector_add(BG->dof,pop);
   
  for( i=0 ; i<BG->k ; i++ ){
	a = fff_vector_get(BG->prior_means_scale,i);
	for (j=0 ; j<BG->dim ; j++){
	  x = 1/fff_matrix_get(BG->prior_precisions,i,j);
	  x = x + fff_matrix_get(variance,i,j);
	  dx = fff_matrix_get(BG->means,i,j)-fff_matrix_get(BG->prior_means,i,j);
	  x = x + a*dx*dx;
	  x = 1./x;
	  fff_matrix_set(BG->precisions,i,j,x);
	}
  }
  
  
  fff_matrix_delete(means);
  fff_matrix_delete(variance);
  fff_vector_delete(pop);
  fff_array_delete(choice);
  
  return(LL);
}

extern int fff_BGMM_Gibbs_sampling(fff_vector* density, fff_Bayesian_GMM* BG, const fff_matrix *X, const fff_matrix *grid, const int niter, const int method)
{
  double LL;
  fff_matrix * proba = fff_matrix_new(grid->size1,BG->k);
  fff_vector *v = fff_vector_new(grid->size1);
  /* it is assumed here that the MC is stationary */
  int i,j;

   for (i=0 ; i<niter ; i++){
	 _fff_update_BGMM(BG,X,i+niter,method);
	 if (method == 0) LL = _fff_Npval(proba,grid,BG);
	 else LL = fff_WNpval(proba,grid,BG);
	 for (j=0; j <BG->k;j++){
	   fff_matrix_get_col(v,proba,j);
	   fff_vector_add(density,v);
	 }
   }
   fff_vector_scale(density,1./niter);

   fff_vector_delete(v);
   fff_matrix_delete(proba);
   return 0;
}

extern int fff_BGMM_Gibbs_estimation(fff_matrix* membership, fff_Bayesian_GMM* BG, const fff_matrix *X, const int niter, const int method)
{
  int i=0;
  double LL;
  _fff_BGMM_init(BG);
  
  fff_matrix_set_all(membership,0);
  fff_matrix * average_means = fff_matrix_new(BG->k,BG->dim);
  fff_matrix * average_precisions = fff_matrix_new(BG->k,BG->dim);
  fff_vector * average_means_scale = fff_vector_new(BG->k);
  fff_vector * average_dof =  fff_vector_new(BG->k);
  fff_vector * average_weights =  fff_vector_new(BG->k);
  fff_matrix * proba;
  
  
  /* burn-in period */
  for (i=0 ; i<niter ; i++)
	LL = _fff_update_BGMM(BG,X,i,method);
  
  /* final updates */
  proba = fff_matrix_new(X->size1,BG->k);
  
  for (i=0 ; i<niter ; i++){
	_fff_full_update_BGMM(proba,BG,X,i+niter,method);
	
	fff_matrix_add(membership,proba);
	fff_matrix_add(average_means,BG->means);
	fff_matrix_add(average_precisions,BG->precisions);
	fff_vector_add(average_means_scale,BG->means_scale);
	fff_vector_add(average_dof,BG->dof );
	fff_vector_add(average_weights,BG->weights );
	
  }
  
  fff_matrix_scale(membership,1./niter);
  fff_matrix_scale(average_means,1./niter);
  fff_matrix_scale(average_precisions,1./niter);
  fff_vector_scale(average_means_scale,1./niter);
  fff_vector_scale(average_dof,1./niter);
  fff_vector_scale(average_weights,1./niter);

  fff_matrix_memcpy(BG->means,average_means);
  fff_matrix_memcpy(BG->precisions,average_precisions);
  fff_vector_memcpy(BG->means_scale,average_means_scale);
  fff_vector_memcpy(BG->dof,average_dof);
  fff_vector_memcpy(BG->weights,average_weights);
 
  fff_matrix_delete(proba);
  
  return(0);
}

extern int fff_BGMM_get_model( fff_matrix * means, fff_vector * means_scale,  fff_matrix * precisions, fff_vector* dof, fff_vector * weights, const fff_Bayesian_GMM* BG)
{
  fff_matrix_memcpy(means,BG->means);
  fff_matrix_memcpy(precisions,BG->precisions);
  fff_vector_memcpy(means_scale,BG->means_scale);
  fff_vector_memcpy(dof,BG->dof);
  fff_vector_memcpy(weights,BG->weights);
  
  return(0);
}


extern int fff_BGMM_set_model( fff_Bayesian_GMM* BG, const fff_matrix * means, const fff_vector * means_scale,const  fff_matrix * precisions, const fff_vector* dof, const fff_vector * weights)
{
  
  fff_matrix_memcpy(BG->means,means);
  fff_matrix_memcpy(BG->precisions,precisions);
  fff_vector_memcpy(BG->means_scale,means_scale);
  fff_vector_memcpy(BG->dof,dof);
  fff_vector_memcpy(BG->weights,weights);
  
  return(0);
}

