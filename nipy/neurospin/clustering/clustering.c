#include "fffpy.h"
#include <fff_clustering.h>
#include <fff_graphlib.h>

#include <fff_GMM.h>
#include <fff_DPMM.h>
/*
#include <fff_HGMM.h>  
#include <fff_constrained_clustering.h>
*/
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API



/* Code pour la creation du module */ 
static char ward_doc[]=
" parent,cost = ward(X) \n\
ward clustering algorithm\n\
INPUT:\n\
- X data array with shape(n,p)\n\
OUPUT:\n\
- parent: array of shape (2*n-1) \n\
that represents the tree structure \n\
associated with the hierrachical clustering \n\
- cost: arrao of shape (2*n-1)\n\
that represents the cost value associated with each cluster\n\
(the first n values are zero)";

static char cmeans_doc[] =
" Centers, Labels, J = cmeans(X,nbclusters,Labels,maxiter,delta)\n\
  cmeans clustering algorithm \n\
 INPUT : \n\
 - A data array X, supposed to be written as (n*p)\n\
   where n = number of features, p =number of dimensions\n\
 - nbclusters (int), the number of desired clusters\n\
 - Labels n array of predefined Labels. Default value is NULL.\n\
   if NULL or if Labels->size != X->size1, a random initilization is performed.\n\
 - maxiter(int, =300 by default), the maximum number \n\
   of iterations  before convergence\n\
 - delta(double, =0.0001 by default), \n\
  the relative increment in the results before declaring convergence\n\
 OUPUT :\n\
 - Centers: array of size nbclusters*p, the centroids of \n\
  the resulting clusters\n\
 - Labels : arroy of size n, the discrete labels of the input items;\n\
 - J the final value of the criterion";

static char voronoi_doc[] = 
" Labels = voronoi(X,Centers)\n\
  Voronoi (nearest neighbour) assignment of the data to centers \n\
 INPUT :\n\
 - A data array X, supposed to be written as (n*p)\n\
  where n = number of features, p =number of dimensions\n\
 - Centers: array of size nbclusters*p, the centroids of \n\
  the clusters\n\
 OUPUT :\n\
 - Labels : arroy of size n, the discrete labels of the input items \n\
\n\
Note that the implementation is naive";

static char fcm_doc[] = 
"(Centers, Labels) = fcm(X,nbclusters,maxiter,delta)\n\
  fuzzy cmeans clustering algorithm\n\
 INPUT :\n\
 -	A data array X, supposed to be written as (n*p)\n\
	where n = number of features, p =number of dimensions\n\
 - nbclusters (int), the number of desired clusters\n\
 - maxiter(int, =300 by default), the maximum number of iterations before convergence\n\
 - delta(double, =0.0001 by default), the relative increment in the results before declaring convergence\n\
 OUPUT :\n\
 - Centers: array of size nbclusters*p, the centroids of the resulting clusters\n\
 - Labels : arroy of size n, the discrete labels of the input items\n\
Note that the fuzzy index is forced to be 2";

/*
static char constrained_cmeans_doc[] = 
" Centers, SideC, Labels, J = constrained_cmeans(X,SideX, nbclusters,dmax,Labels,maxiter,delta)\n\
  cmeans clustering algorithm constrained by side information \n\
 INPUT :\n\
 - A data array X, addumed to be written as (n*p)\n\
   where n = number of features, p =number of dimensions\n\
- A data array SideX, addumed to be written as (n*s)\n\
   where n = number of features, s =number of dimensions\n\
- nbclusters (int), the number of desired clusters\n\
-dmax (double) is the threshold for admissible assignments \n\
 - Labels n array of predefined Labels. Default value is None.\n\
   if None or iappropriate, a random initilization is performed.\n\
 - maxiter(int, =300 by default), the maximum number \n\
   of iterations  before convergence\n\
 - delta(double, =0.0001 by default), \n\
  the relative increment in the results before declaring convergence\n\
 OUPUT :\n\
 - Centers: array of size nbclusters*p, the centroids of \n\
  the resulting clusters\n\
- SideC: array of size nbclusters*s, the centroids of \n\
  the resulting clusters in the space of side information \n\
 - Labels : arroy of size n, the discrete labels of the input items;\n\
 - J the final value of the criterion";

static char constrained_match_doc[] = 
"  matches, J = constrained_match(Centers, SideC,X,SideX,dmax)\n\
  template matching constrained by side information \n\
 INPUT :\n\
 - Centers: array of size nbclusters*p, the centroids of \n\
  the targets\n\
- SideC: array of size nbclusters*s, the centroids of \n\
  the targets in the space of side information \n\
 - A data array X, addumed to be written as (n*p)\n\
   where n = number of features, p =number of dimensions\n\
- A data array SideX, addumed to be written as (n*s)\n\
   where n = number of features, s =number of dimensions\n\
- dmax (double) is the threshold for admissible assignments \n\
 OUPUT :\n\
 - matches : arroy of size (nbclusters), the discrete labels of the \n\
best matching data, given the template\n\
 - J the corresponding stress criterion" ;
*/
static char gmm_doc[] = 
  
" Centers, Precision, Weights, Labels, LogLike = gmm(X,nbclusters,Labels,prec_type,maxiter,delta)\n\
  Gaussian Mixture Model (GMM) clustering algorithm \n\
 INPUT :\n\
 -	A data array X, supposed to be written as (n*p)\n\
	where n = number of features, p =number of dimensions\n\
 - nbclusters (int), the number of desired clusters\n\
 - Labels n array of predefined Labels. Default value is null.\n\
   if null or iappropriate, a random initilization is perfomed.\n\
 - prec_type(int,=1 by default) yields the type of \n\
   covariance/precision matrice used \n\
   prec_type==0 -> full covariance matrices\n\
   prec_type==1 -> diagonal covariance matrices\n\
   prec_type==2 -> the same diagonal matrix for all clusters \n\
 - maxiter(int, =300 by default), the maximum number \n\
   of iterations  before convergence\n\
 - delta(double, =0.0001 by default), \n\
  the relative increment in the results before declaring convergence\n\
 OUPUT :\n\
 - Centers: array of size nbclusters*p, the centroids of \n\
  the resulting clusters\n\
 - Precision: the precision matrices of the clusters\n\
 - Weights: the weight of the mixture \n\
 - Labels : arroy of size n, the discrete labels of the input items\n\
 - LogLike : (scalar)average Log-Likelihood of the data for the GMM model";

static char gmm_relax_doc[] = 
  
" Centers, Precision, Weights, Labels, LogLike = gmm_relax(X,Centers, Precision, Weights,maxiter,delta)\n\
  Relaxing of the  GMM on the dataset X\n\
 INPUT :\n\
 -	A data array X, supposed to be written as (n*p)\n\
	where n = number of features, p =number of dimensions \n\
 - Centers: array of size nbclusters*p, the centroids of \n\
  the clusters\n\
 - Precision: the precision matrices of the clusters\n\
 - Weights: the weight of the mixture \n\
 - maxiter(int, =300 by default), the maximum number \n\
   of iterations  before convergence\n\
 - delta(double, =0.0001 by default), \n\
  the relative increment in the results before declaring convergence\n\
 OUPUT :\n\
 - Centers: array of size nbclusters*p, the centroids of \n\
  the resulting clusters\n\
 - Precision: the precision matrices of the clusters\n\
 - Weights: the weight of the mixture \n\
 - Labels : array of size n, the discrete labels of the input items\n\
 - LogLike : array of size n, the Log-Likelihood of the data for the GMM model";

static char gmm_partition_doc[] = 
" LogLike, Labels = gmm_partition(X,Centers,Precision, Weights)\n\
  Fits Gaussian Mixture Model (GMM) to the data X \n\
 INPUT :\n\
 -	A data array X, supposed to be written as (n*p)\n\
	where n = number of features, p =number of dimensions\n\
 - Centers: array of size nbclusters*p, the centroids of \n\
  the clusters\n\
 - Precision: array of size nbclusters*p, the precision of \n\
  the  clusters\n\
 OUPUT :\n\
 - Labels : arroy of size n, the discrete labels of the input items\n\
 - LogLike : array of size n, the log-likelihood of the items \n\
	with respect to the model.";

static char gmm_membership_doc[] = 
" a,b,d = gmm_membserhsip(X,Centers,Precision, Weights)\n\
  Gets membership of X with respect to the GMM \n\
 INPUT :\n\
 -	A data array X, supposed to be written as (n*p)\n\
	where n = number of features, p =number of dimensions\n\
 - Centers: array of size nbclusters*p, the centroids of \n\
  the clusters\n\
 - Precision: array of size nbclusters*p, the precision of \n\
  the  clusters\n\
 OUPUT :\n\
 - A,B,D: three vectors that define the  birpartite membership graph";

static char gmm_shift_doc[] = 
" X = gmm_membserhsip(X,Centers,Precision, Weights)\n\
  Shidts X to increase its likelihood within the GMM \n\
 INPUT :\n\
 -	A data array X, supposed to be written as (n*p)\n\
	where n = number of features, p =number of dimensions\n\
 - Centers: array of size nbclusters*p, the centroids of \n\
  the clusters\n\
 - Precision: array of size nbclusters*p, the precision of \n\
  the  clusters\n\
 OUPUT :\n\
 - X: The shifted data";

static char bayesian_gmm_doc[] = 
" membership, mean, mean_scale, precision_scale, weights,dof,density = gibbs_gmm(X,prior_centers,prior_precision, prior_mean_scale, prior_weights, prior_dof, niter=1000,delta=0.0001,grid = None)\n\
  Variational Bayes Gaussian Mixture Model (GMM) clustering algorithm \n\
INPUT :\n\
  -	A data array X, supposed to be written as (n*p)\n\
	where n = number of features, p =number of dimensions\n\
 - prior_centers is a (nbclusters,p) array \n\
   that contains the prior cluster centers \n\
 - prior precision is an (nbclusters,p) array \n\
   that contains the diagonal prior precision of the clsuters \n\
 - prior_mean_scale is an (nbclusters) array \n\
   it gives the scaling factor on the precision of the mean of each cluster \n\
 - prior_weights is an (nbclusters) array \n\
   it gives the dirichlet priors on the component weights \n\
 - dof is an (nbclusters) array \n\
  it gives the prior dofs on teh precision matrix \n\
 - niter=1000, the number of iterations in the VB algo \n\
 - delta(double, =0.0001 by default), \n\
  the relative increment in the results before declaring convergence\n\
 - grid=None sampling grid. by default, it is X \n\
 OUPUT :\n\
 - Labels : arroy of size n, the discrete labels of the input items\n\
 - mean: array of size nbclusters*p, the posterior means \n\
 - mean_scale: array of size nbclusters, the posterior scaling factors \n\
 - Precision_scale : array of size nbclusters*p, the posterior precisions matrices \n\
 - Weights: array of size nbclusters, the posterior weights of the mixture \n\
 - dof : arroy of size nbclusters, \n\
the posterior  degrees of freedom  of the precisions.\n\
 - density : Density of the data on the sampling grid.";

static char bayesian_gmm_sampling_doc[] = 
" density = gibbs_gmm(prior_centers,prior_precision, prior_mean_scale, prior_weights, prior_dofmean, mean_scale, precision_scale, weights,dof,grid)\n\
  Variational Bayes Gaussian Mixture Model (GMM) clustering algorithm \n\
INPUT :\n\
 - prior_centers is a (nbclusters,p) array \n\
   that contains the prior cluster centers \n\
 - prior precision is an (nbclusters,p) array \n\
   that contains the diagonal prior precision of the clsuters \n\
 - prior_mean_scale is an (nbclusters) array \n\
   it gives the scaling factor on the precision of the mean of each cluster \n\
 - prior_weights is an (nbclusters) array \n\
   it gives the dirichlet priors on the component weights \n\
 - prior_dof is an (nbclusters) array \n\
  it gives the prior dofs on teh precision matrix \n\
  - mean: array of size nbclusters*p, the posterior means \n\
- mean_scale: array of size nbclusters, the posterior scaling factors \n \
 - Precision_scale : array of size nbclusters*p, the posterior precisions matrices \n\
 - Weights: array of size nbclusters, the posterior weights of the mixture \n\
 - dof : arroy of size nbclusters, \n\
the posterior  degrees of freedom  of the precisions.\n\
 - grid=None sampling grid. \n\
 OUPUT :\n\
 - density : Density of the data on the sampling grid.";

static char gibbs_gmm_doc[] = 
" membership, mean, mean_scale, precision_scale, weights,dof,density = gibbs_gmm(X,prior_centers,prior_precision, prior_mean_scale, prior_weights, prior_dof, niter=1000,method=1,grid = None,nsamplings = 1)\n\
  MCMC Bayesian Gaussian Mixture Model (GMM) clustering algorithm \n\
This is a based on a conjugate Wishart-Normal prior model. \n\
Moreover, covariance/precision matrices are restricted to be diagonal \n\
 INPUT :\n\
 -	A data array X, supposed to be written as (n*p)\n\
	where n = number of features, p =number of dimensions\n\
 - prior_centers is a (nbclusters,p) array \n\
   that contains the prior cluster centers \n\
 - prior precision is an (nbclusters,p) array \n\
   that contains the diagonal prior precision of the clsuters \n\
 - prior_mean_scale is an (nbclusters) array \n\
   it gives the scaling factor on the precision of the mean of each cluster \n\
 - prior_weights is an (nbclusters) array \n\
   it gives the dirichlet priors on the component weights \n\
 - dof is an (nbclusters) array \n\
  it gives the prior dofs on teh precision matrix \n\
 - niter=1000, the number of iterations in the MCMC sampling \n\
 - method=1 the method used to derive the posterior probability \n\
  if method==0, a gaussian  model is used, i.e. precisions are fixed\n\
 - grid = None is a sampling for the posterior \n\
   by default grid = X \n\
 - nsamples = 1 is the number of samplings for averaging the posterior \n\
 OUPUT :\n\
 - membership: array of size n*nbclusters, \n\
the relative probabilities of the membership \n\
 - mean: array of size nbclusters*p, the posterior means \n\
 - mean_scale: array of size nbclusters, the posterior scaling factors \n\
 - Precision_scale : array of size nbclusters*p, the posterior precisions matrices \n\
 - Weights: array of size nbclusters, the posterior weights of the mixture \n\
 - dof : arroy of size nbclusters, \n\
the posterior  degrees of freedom  of the precisions \n\
 - density: array of shape shape(X,0) or shape(grid,0) if supplied  \n\
the predictive density obtained by averaging across iterations";


static char fdp_doc[] = 
" density,posterior = fdp(X, alpha, g0,g1,prior_dof,precisions, pvals, labels, niter=1000,grid=None,nis=1,nii=1)\n\
  Customized Dirichlet Process Mixture Model clustering algorithm \n\
  The DPMM is customized in the sense that \n\
 - its components are Gaussian with fixed diaginal covariance \n\
 - there is additionall a null class and a p-value that the data belongs to the null class \n\
 - the base distribution G is uniform over a certain volume \n\
  INPUT:\n\
  - X: the input data, an array of shape (nbitems,dim)\n\
  - alpha (float): the cluster creation parameter in the DP \n\
  - g0 (float): the constant of the uniform distribution (used for H0) \n\
  - g1 (float): idem g0, but used for H1 \n\
  - prior_dof: number of dof of the prior covariance/precision model \n\
    if prior_dof==0, the model behaves as if prior_dof=infty \n\
    i.e. the posterior precision is equal to the prior. \n\
  - precisions : fixed cluster diagonal precision, defined as (1,dim) array \n\
  - pvals: prior probability of H1 for the input data. size=nbitems \n\
  - labels : chunking labels: can be either [0:nbitems] or contain some =information (e.g. the subjects from which the data is sampled) \n\
  - niter=1000: the number of ietrations for the burn-in period \n\
  - grid: the set of nodes for which the likelihood of the model under H1 is computed ; should have size (nbnodes, dim) \n\
  - nis =1 : number of iterations where the likelhood is sampled on the grid \n\
  - nii=1 :number of iterationsfor the computation of the posterior probability of H1 given the data \n\
  OUPUT: \n\
  - density: the likelihood of the H1 model on the grid (shape=(nbnodes))\n\
  - posterior: the posterior probability of H1 for the inputa data \n\
";

static char dpmm_doc[] = 
" density = dpmm(X, alpha, precisions, prior_means, prior_mean_scale, labels, niter=1000,grid=None,nis=1, dof=0)\n\
  Gaussian Dirichlet Process Mixture Model clustering algorithm \n\
  where components are Gaussian with fixed diaginal covariance \n\
  INPUT:\n\
  - X: the input data, an array of shape (nbitems,dim)\n\
  - alpha (float): the cluster creation parameter in the DP \n\
  - prio_means : fixed cluster diagonal precision, defined as (1,dim) array \n \
  - precisions : fixed cluster diagonal precision, defined as (1,dim) array \n \
  - prior_mean_scale : fixed cluster diagonal precision, defined as (1,dim) array \n\
  - labels : chunking labels: can be either [0:nbitems] or contain some =information (e.g. the subjects from which the data is sampled) \n\
  - niter=1000: the number of ietrations for the burn-in period \n\
  - grid: the set of nodes for which the likelihood of the model under H1 is computed ; should have size (nbnodes, dim) \n\
  - nis =1 : number of iterations where the likelhood is sampled on the grid \n\
  - dof = 0 : is the prior dof of the components \n\
  if dof==0, a model with fixed precision is used. \n\
  OUPUT: \n\
  - density: the posterior proba of the model on the grid (shape=(nbnodes))\n\
"; 
/* 
static char hgmm_doc[] = 
" hgmm(x,nbclusters,labels,subjects, prior_prec,prior_dofs,maxiter,nbsubj,alpha); ";
*/
static char module_doc[] =
" Clustering routines.\n\
Author: Bertrand Thirion (INRIA Saclay, Orsay, France), 2004-2008.";

static PyObject* ward(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *cost, *parent ;
  int n,q;

  int OK = PyArg_ParseTuple( args, "O!:ward", 
							 &PyArray_Type, &x);
	
    if (!OK) Py_RETURN_NONE;
  
    /* prepare C arguments */ 
	
	fff_matrix* X = fff_matrix_fromPyArray( x );
	n = X->size1;
	q = 2*n-1;
	
  	
	fff_array *Parent = fff_array_new1d(FFF_LONG,q);
	fff_vector *Cost = fff_vector_new(q);
	fff_vector_set_all(Cost,0);
	
	fff_clustering_ward(Parent,Cost,X);
	
	/* get the results as python arrrays */
	cost = fff_vector_toPyArray( Cost ); 
	parent = fff_array_toPyArray( Parent );
	fff_matrix_delete(X);
	

	/* Output tuple */ 
	PyObject *ret = Py_BuildValue("NN",parent,cost);
	return ret;
}

static PyObject* cmeans(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *centers, *labels ;
 
  int maxiter = 30;
  double delta = 0.0001;
  int nbclusters; 
  fff_array* Label; 
  labels = NULL;
  
  /* Parse input */ 
	/* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
	
  int OK = PyArg_ParseTuple( args, "O!i|O!id:cmeans", 
			  &PyArray_Type, &x, 
			  &nbclusters,
			  &PyArray_Type, &labels, 
			  &maxiter, 
              &delta ); 

    if (!OK) Py_RETURN_NONE;
    if (nbclusters<1)  Py_RETURN_NONE; 
	
    /* prepare C arguments */ 
	
	fff_matrix* X = fff_matrix_fromPyArray( x ); 
  	fff_matrix* Centers = fff_matrix_new( nbclusters, X->size2 ); 
 	 
	if (labels==NULL)
	  Label = fff_array_new1d(FFF_LONG, X->size1 );
	else{
	  Label = fff_array_fromPyArray( labels ); 
	  if (Label->dimX != X->size1){
		fff_array_delete(Label);
		Label = fff_array_new1d(FFF_LONG, X->size1 );
	  }
	}
	
	/* do the job */
	double J =0;	
	J = fff_clustering_cmeans( Centers, Label, X, maxiter, delta );
	
	/* get the results as python arrrays */
	centers = fff_matrix_toPyArray( Centers ); 
	labels = fff_array_toPyArray( Label );
	fff_matrix_delete(X);
	

	/* Output tuple */ 
	PyObject *ret = Py_BuildValue("NNf",centers, labels,J);
	return ret;
	
	Py_RETURN_NONE;
}


static PyArrayObject* voronoi(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *centers, *labels ;
  
  /* Parse input */ 
	/* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!:voronoi", 
			  &PyArray_Type, &x, 
			  &PyArray_Type, &centers); 
  if (!OK) return NULL; 

  /* prepare C arguments */
  fff_matrix* X = fff_matrix_fromPyArray( x ); 
  fff_matrix* Centers =  fff_matrix_fromPyArray( centers ); 
  fff_array *Label = fff_array_new1d(FFF_LONG, X->size1 );

  /* do the job */
  fff_clustering_Voronoi( Label, Centers, X );

  /* get the results as python arrrays*/
  labels = fff_array_toPyArray( Label ); 
  fff_matrix_delete(X);
  fff_matrix_delete(Centers);
  /* Output tuple */
  return labels;
}

static PyObject* fcm(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *centers, *labels ;

  int maxiter = 300;
  double delta = 0.0001;
  int nbclusters ;  

  /* Parse input */ 
	/* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!i|id:fcm", 
			  &PyArray_Type, &x, 
			  &nbclusters, 
              &maxiter, 
              &delta );  
    if (!OK) Py_RETURN_NONE; 

  /* prepare C arguments */
  fff_matrix* X = fff_matrix_fromPyArray( x ); 
  fff_matrix* Centers = fff_matrix_new( nbclusters, X->size2 );  
  fff_array* Label = fff_array_new1d(FFF_LONG, X->size1 ); 
	
  /* do the job */
  fff_clustering_fcm( Centers, Label, X, maxiter, delta );
  
  fff_matrix_delete(X);
  /* get the results as python arrrays*/
  centers = fff_matrix_toPyArray( Centers ); 
  labels = fff_array_toPyArray( Label ); 
  
  /* Output tuple */

  PyObject *ret = Py_BuildValue("NN",centers,labels);
  return ret;
}
/*
static PyObject* constrained_cmeans(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *sx, *centers,*sc, *labels ;

  int maxiter = 30;
  double eps,delta = 0.0001;
  int nbclusters; 
  fff_array* Label;
  labels = NULL;
  
  int OK = PyArg_ParseTuple( args, "O!O!id|O!id:constrained_cmeans", 
			  &PyArray_Type, &x, 
			  &PyArray_Type, &sx, 
			  &nbclusters,
			  &eps,
			  &PyArray_Type, &labels, 
			  &maxiter, 
                          &delta ); 
    if (!OK) Py_RETURN_NONE;
    if (nbclusters<1)  Py_RETURN_NONE;
 
  fff_matrix* X = fff_matrix_fromPyArray( x ); 
  fff_matrix* SideX = fff_matrix_fromPyArray( sx ); 
  fff_matrix* Centers = fff_matrix_new( nbclusters, X->size2 ); 
  fff_matrix* SideC = fff_matrix_new( nbclusters, SideX->size2 ); 

  if (labels==NULL){
	Label = fff_array_new1d(FFF_LONG, X->size1 );
  }
  else
    Label = fff_array_fromPyArray( labels ); 
  
  
  double J = 0;
  J = fff_constrained_cmeans(Centers,SideC,Label,X,SideX,eps,maxiter,delta);
	       
  centers = fff_matrix_toPyArray( Centers ); 
  sc = fff_matrix_toPyArray( SideC ); 
  labels = fff_array_toPyArray( Label ); 
  fff_matrix_delete(X);
  fff_matrix_delete(SideX);

  PyObject *ret = Py_BuildValue("NNNd",centers,sc, labels,PyFloat_FromDouble(J));
  return ret;
}

static PyObject* constrained_match(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *sx, *centers,*sc, *matches ;

  double eps;
  
  int OK = PyArg_ParseTuple( args, "O!O!O!O!d:constrained_match", 
			  &PyArray_Type, &centers, 
			  &PyArray_Type, &sc, 
			  &PyArray_Type, &x, 
			  &PyArray_Type, &sx, 
                          &eps ); 
    if (!OK) Py_RETURN_NONE;
    
  fff_matrix* X = fff_matrix_fromPyArray( x ); 
  fff_matrix* SideX = fff_matrix_fromPyArray( sx ); 
  fff_matrix* Centers = fff_matrix_fromPyArray( centers ); 
  fff_matrix* SideC = fff_matrix_fromPyArray( sc );

  fff_array* Matches = fff_array_new1d(FFF_LONG, Centers->size1 );
  
  double J = 0;
  
  J = fff_constrained_match(Matches, Centers, SideC, X, SideX, eps);
  
  matches = fff_array_toPyArray( Matches ); 
  
  fff_matrix_delete(X);
  fff_matrix_delete(SideX);
  fff_matrix_delete(Centers);
  fff_matrix_delete(SideC);
  
  PyObject *ret = Py_BuildValue("Nd",matches,PyFloat_FromDouble(J));
  return ret;
}
*/

static PyObject* gmm(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *centers, *labels, *precision, *weights ;

  int maxiter = 300;
  double delta = 0.0001; 
  int prec_type = 1; 
  int nbclusters;
  int chunksize = 0;
  fff_array* Label;
  labels = NULL;
  int verbose = 0;
  
  int OK = PyArg_ParseTuple( args, "O!i|O!iidii:gmm", 
							 &PyArray_Type, &x, 
							 &nbclusters,
							 &PyArray_Type, &labels, 
							 &prec_type,
							 &maxiter, 
							 &delta,
							 &chunksize,
							 &verbose
							 ); 
  if (!OK) Py_RETURN_NONE; 
  
  fff_matrix* X = fff_matrix_fromPyArray( x ); 
  fff_matrix* Centers = fff_matrix_new( nbclusters, X->size2 );  
  fff_vector* Weights = fff_vector_new( nbclusters );
  fff_matrix* Precision = NULL;
  int fd = (int) X->size2;

  
  if (chunksize<nbclusters)
    chunksize = 1000000;
  if (chunksize>X->size1)
    chunksize = X->size1;

 
  switch (prec_type){
	  case 0:{
	    Precision = fff_matrix_new( nbclusters, fd*fd );
	    break;
	  }
	  case 2:{
	    Precision = fff_matrix_new( 1, fd );
	    break;
	  }
	  case 1:{
	      Precision = fff_matrix_new( nbclusters, fd );
	      break;
	    } 
	  } 
 
  if (labels==NULL){
	Label = fff_array_new1d(FFF_LONG, X->size1 );
	}  
  else
	Label = fff_array_fromPyArray( labels ); 
  
  double J = 0;
  
  J = fff_clustering_gmm( Centers, Precision, Weights, Label, X, maxiter, delta, chunksize, verbose );

  fff_matrix_delete(X);
  centers = fff_matrix_toPyArray( Centers ); 
  labels = fff_array_toPyArray( Label ); 
  precision = fff_matrix_toPyArray( Precision );
  weights = fff_vector_toPyArray( Weights ); 

	PyObject *ret = Py_BuildValue("NNNNd",centers,precision,weights, labels,J);
  return ret;
}

static PyObject* gmm_relax(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *centers, *labels, *precision, *weights, *loglike ;

  int maxiter = 300;
  double delta = 0.0001;
  fff_array* Label;
  labels = NULL;
  
  int OK = PyArg_ParseTuple( args, "O!O!O!O!|id:gmm", 
							 &PyArray_Type, &x, 
							 &PyArray_Type, &centers, 
							 &PyArray_Type, &precision, 
							 &PyArray_Type, &weights, 
							 &maxiter, 
							 &delta
							 ); 
  if (!OK) Py_RETURN_NONE; 
  
  fff_matrix* X = fff_matrix_fromPyArray( x ); 
  fff_matrix* Centers = fff_matrix_fromPyArray( centers); 
  fff_vector* Weights = fff_vector_fromPyArray(weights);
  fff_matrix* Precision = fff_matrix_fromPyArray( precision ); ;

  Label = fff_array_new1d(FFF_LONG, X->size1 );
  fff_vector * LogLike = fff_vector_new( X->size1 ); 
  
  fff_gmm_relax( LogLike, Label, Centers, Precision, Weights, X, maxiter, delta);

  fff_matrix_delete(X);
  centers = fff_matrix_toPyArray( Centers ); 
  labels = fff_array_toPyArray( Label ); 
  precision = fff_matrix_toPyArray( Precision );
  weights = fff_vector_toPyArray( Weights );
  loglike = fff_vector_toPyArray( LogLike );


  PyObject *ret = Py_BuildValue("NNNNN",centers,precision,weights, labels,loglike);
  return ret;
}

static PyObject* gmm_partition(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *centers, *labels, *precision, *weights, *loglike ;

  fff_array* Label;
  fff_vector *LogLike;
  
  int OK = PyArg_ParseTuple( args, "O!O!O!O!:gmm_partition", 
			  &PyArray_Type, &x, 
			  &PyArray_Type, &centers, 
			  &PyArray_Type, &precision, 
			  &PyArray_Type, &weights ); 
    if (!OK) Py_RETURN_NONE; 

  fff_matrix* X = fff_matrix_fromPyArray( x ); 
  fff_matrix* Centers = fff_matrix_fromPyArray( centers );
  fff_matrix* Precision = fff_matrix_fromPyArray( precision ); 
  fff_vector* Weights = fff_vector_fromPyArray( weights );

  Label = fff_array_new1d(FFF_LONG, X->size1 );
  LogLike = fff_vector_new( X->size1);

  fff_gmm_partition( LogLike, Label, X, Centers, Precision, Weights);
  fff_matrix_delete(X);
  fff_matrix_delete(Centers);
  fff_matrix_delete(Precision);
  fff_vector_delete(Weights);
  
  labels = fff_array_toPyArray( Label ); 
  loglike = fff_vector_toPyArray( LogLike );


  PyObject *ret = Py_BuildValue("NN",labels, loglike);
   
  return ret;
}

static PyObject* gmm_membership(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *centers, *a, *b, *d, *precision, *weights;

  int OK = PyArg_ParseTuple( args, "O!O!O!O!:gmm_membership", 
			  &PyArray_Type, &x, 
			  &PyArray_Type, &centers,
			  &PyArray_Type, &precision, 
			  &PyArray_Type, &weights ); 
    if (!OK) Py_RETURN_NONE; 

  fff_matrix* X = fff_matrix_fromPyArray( x ); 
  fff_matrix* Centers = fff_matrix_fromPyArray( centers );
  fff_matrix* Precision = fff_matrix_fromPyArray( precision );
  fff_vector* Weights = fff_vector_fromPyArray( weights );

  int V = X->size1;
  fff_graph *G = fff_graph_new(V,0);
  
  int E = fff_gmm_membership( G, X, Centers, Precision, Weights);
  fff_graph_reset( &G, V, E);
  E = fff_gmm_membership( G, X, Centers, Precision, Weights);
  
  fff_matrix_delete(X);
  fff_matrix_delete(Centers);
  fff_matrix_delete(Precision);
  fff_vector_delete(Weights);
   
  fff_array *A = fff_array_new1d(FFF_LONG,E);
  fff_array *B = fff_array_new1d(FFF_LONG,E);
  fff_vector *D = fff_vector_new(E);
  fff_graph_edit_safe(A,B,D,G);
  fff_graph_delete(G);
   
  a = fff_array_toPyArray( A );
  b = fff_array_toPyArray( B );
  d = fff_vector_toPyArray( D );  
  PyObject* ret = Py_BuildValue("NNN", 
								a,
								b,
								d); 
  return ret;
}

static PyArrayObject* gmm_shift(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *centers, *precision, *weights;

  int OK = PyArg_ParseTuple( args, "O!O!O!O!:gmm_shift", 
							 &PyArray_Type, &x,
			  &PyArray_Type, &centers,
			  &PyArray_Type, &precision, 
			  &PyArray_Type, &weights ); 
  if (!OK) return NULL; 

  fff_matrix* X = fff_matrix_fromPyArray( x ); 
  fff_matrix* Centers = fff_matrix_fromPyArray( centers );
  fff_matrix* Precision = fff_matrix_fromPyArray( precision );
  fff_vector* Weights = fff_vector_fromPyArray( weights );

  fff_gmm_shift( X, Centers, Precision, Weights);
  fff_matrix_delete(Centers);
  fff_matrix_delete(Precision);
  fff_vector_delete(Weights);
  
  x = fff_matrix_toPyArray( X );

  return x;
}


static PyObject* bayesian_gmm(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *mean, *label, *weights, *prior_centers, *prior_precision, *prior_mean_scale, *prior_dof, *prior_weights, *dof, *mean_scale, *precision_scale, *density;

  int k,dim,niter = 1000;
  double delta = 0.0001;
  PyArrayObject *grid = NULL;
  label = NULL;
  int nsamplings = 1;

  int OK = PyArg_ParseTuple( args, "O!O!O!O!O!O!|O!idO!:bayesian_gmm", 
							 &PyArray_Type, &x, 
							 &PyArray_Type, &prior_centers,
							 &PyArray_Type, &prior_precision,
							 &PyArray_Type, &prior_mean_scale,	 
							 &PyArray_Type, &prior_weights,
							 &PyArray_Type, &prior_dof,  
							 &PyArray_Type, &label,
							 &niter,
							 &delta,
							 &PyArray_Type, &grid
							); 
  if (!OK) Py_RETURN_NONE; 

  fff_matrix* X = fff_matrix_fromPyArray( x );

  fff_matrix *PriorPrecision = fff_matrix_fromPyArray( prior_precision ); 
  fff_matrix *PriorCenters = fff_matrix_fromPyArray( prior_centers);
  fff_vector *PriorMeanScale = fff_vector_fromPyArray( prior_mean_scale);
  fff_vector *PriorDof = fff_vector_fromPyArray(prior_dof);
  fff_vector *PriorWeights = fff_vector_fromPyArray(prior_weights);

  k = PriorCenters->size1; 
  dim = X->size2;	
  fff_array *Label ;
  if (label==NULL)
	Label = fff_array_new1d(FFF_LONG,X->size1);
  else 
	Label = fff_array_fromPyArray(label);

  fff_matrix *Mean = fff_matrix_new(k,dim);
  fff_vector *MeanScale = fff_vector_new(k);
  fff_matrix *PrecisionScale = fff_matrix_new(k,dim);
  fff_vector *Dof = fff_vector_new(k);
  fff_vector *Weights = fff_vector_new(k);

  fff_Bayesian_GMM *BG =  fff_BGMM_new( k,dim );
  
  fff_BGMM_set_priors(BG, PriorCenters, PriorMeanScale, PriorPrecision, PriorDof, PriorWeights);  
  
  
  fff_VB_gmm_estimate (BG, Label, X, niter, delta);
  
  fff_BGMM_get_model( Mean, MeanScale,PrecisionScale, Dof, Weights, BG);
  
  fff_matrix* Grid;
  if (grid==NULL){
	Grid = X;
  }
  else{
	Grid = fff_matrix_fromPyArray( grid );
  }
  fff_vector *Density = fff_vector_new(Grid->size1);
  if (nsamplings>0)
	fff_BGMM_sampling(Density, BG,Grid);
  
  density = fff_vector_toPyArray( Density);
  
  if (grid !=NULL)
	fff_matrix_delete(Grid);
  fff_BGMM_delete( BG );

  fff_matrix_delete(X);

  fff_matrix_delete(PriorPrecision);
  fff_matrix_delete(PriorCenters);
  fff_vector_delete(PriorMeanScale);
  fff_vector_delete(PriorDof);
  fff_vector_delete(PriorWeights);

  mean = fff_matrix_toPyArray( Mean);
  mean_scale = fff_vector_toPyArray( MeanScale); 
  label = fff_array_toPyArray( Label ); 
  precision_scale = fff_matrix_toPyArray( PrecisionScale );
  weights = fff_vector_toPyArray( Weights );
  dof = fff_vector_toPyArray( Dof ); 

  PyObject *ret = Py_BuildValue("NNNNNNN",label, mean, mean_scale, precision_scale, weights,dof,density );
  return ret;
}

static PyObject* bayesian_gmm_sampling(PyObject* self, PyObject* args)
{ 
  PyArrayObject *mean, *weights, *prior_centers, *prior_precision, *prior_mean_scale, *prior_dof, *prior_weights, *dof, *mean_scale, *precision_scale, *density;

  int k,dim;
  PyArrayObject *grid = NULL;

  int OK = PyArg_ParseTuple( args, "O!O!O!O!O!O!O!O!O!O!O!:bayesian_gmm_sampling", 
							 &PyArray_Type, &prior_centers,
							 &PyArray_Type, &prior_precision,
							 &PyArray_Type, &prior_mean_scale,	 
							 &PyArray_Type, &prior_weights,
							 &PyArray_Type, &prior_dof,
							 &PyArray_Type, &mean,
							 &PyArray_Type, &precision_scale,
							 &PyArray_Type, &mean_scale,	 
							 &PyArray_Type, &weights,
							 &PyArray_Type, &dof,
							 &PyArray_Type, &grid 
							); 
  if (!OK) Py_RETURN_NONE; 
 
 
  fff_matrix *PriorPrecision = fff_matrix_fromPyArray( prior_precision ); 
  fff_matrix *PriorCenters = fff_matrix_fromPyArray( prior_centers);
  fff_vector *PriorMeanScale = fff_vector_fromPyArray( prior_mean_scale);
  fff_vector *PriorDof = fff_vector_fromPyArray(prior_dof);
  fff_vector *PriorWeights = fff_vector_fromPyArray(prior_weights);
  //
  fff_matrix *PrecisionScale = fff_matrix_fromPyArray( precision_scale ); 
  fff_matrix *Mean = fff_matrix_fromPyArray( mean);
  fff_vector *MeanScale = fff_vector_fromPyArray( mean_scale);
  fff_vector *Dof = fff_vector_fromPyArray(dof);
  fff_vector *Weights = fff_vector_fromPyArray(weights);

  k = Mean->size1;
  dim = Mean->size2;	

  fff_Bayesian_GMM *BG =  fff_BGMM_new( k,dim );
  
  fff_BGMM_set_priors(BG, PriorCenters, PriorMeanScale, PriorPrecision, PriorDof, PriorWeights);  
  fff_BGMM_set_model(BG, Mean, MeanScale, PrecisionScale, Dof, Weights);  
  
  fff_matrix* Grid;
  Grid = fff_matrix_fromPyArray( grid );

  fff_matrix *Density = fff_matrix_new(Grid->size1,k);
  fff_WNpval(Density, Grid, BG);
  density = fff_matrix_toPyArray( Density);
  
  
  fff_matrix_delete(Grid);
  fff_BGMM_delete( BG ); 

  fff_matrix_delete(PriorPrecision);
  fff_matrix_delete(PriorCenters);
  fff_vector_delete(PriorMeanScale);
  fff_vector_delete(PriorDof); 
  fff_vector_delete(PriorWeights);

  fff_matrix_delete(Mean); 
  fff_matrix_delete(PrecisionScale);
  fff_vector_delete(MeanScale);
  fff_vector_delete(Dof); 
  fff_vector_delete(Weights); 

  PyObject *ret = Py_BuildValue("N",density );
  return ret; 
}

static PyObject* gibbs_gmm(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *mean, *membership, *weights, *prior_centers, *prior_precision, *prior_mean_scale, *prior_dof, *prior_weights, *dof, *mean_scale, *precision_scale, *grid, *density;

  int k,dim,niter = 1000;
  int method = 1;
  grid = NULL;
  int nsamplings = 0;
  
  int OK = PyArg_ParseTuple( args, "O!O!O!O!O!O!|iiO!i:gibbs_gmm", 
							 &PyArray_Type, &x, 
							 &PyArray_Type, &prior_centers,
							 &PyArray_Type, &prior_precision,
							 &PyArray_Type, &prior_mean_scale,	 
							 &PyArray_Type, &prior_weights,
							 &PyArray_Type, &prior_dof,
							 &niter,
							 &method,
							 &PyArray_Type, &grid,
							 &nsamplings
							); 
  if (!OK) Py_RETURN_NONE; 
 
  fff_matrix* X = fff_matrix_fromPyArray( x ); 

  fff_matrix *PriorPrecision = fff_matrix_fromPyArray( prior_precision ); 
  fff_matrix *PriorCenters = fff_matrix_fromPyArray( prior_centers);
  fff_vector *PriorMeanScale = fff_vector_fromPyArray( prior_mean_scale);
  fff_vector *PriorDof = fff_vector_fromPyArray(prior_dof);
  fff_vector *PriorWeights = fff_vector_fromPyArray(prior_weights);

  k = PriorCenters->size1;
  dim = X->size2;
  fff_matrix *Membership = fff_matrix_new(X->size1,k);

  
  fff_matrix *Mean = fff_matrix_new(k,dim);
  fff_vector *MeanScale = fff_vector_new(k);
  fff_matrix *PrecisionScale = fff_matrix_new(k,dim);
  fff_vector *Dof = fff_vector_new(k);
  fff_vector *Weights = fff_vector_new(k);

  fff_Bayesian_GMM *BG =  fff_BGMM_new( k,dim );
  
  fff_BGMM_set_priors(BG, PriorCenters, PriorMeanScale, PriorPrecision, PriorDof, PriorWeights);  
  
  fff_BGMM_Gibbs_estimation(Membership,BG, X, niter,method);
  
  fff_BGMM_get_model( Mean, MeanScale,PrecisionScale, Dof, Weights, BG);
  
  fff_matrix* Grid;
  if (grid==NULL){
	Grid = X;
  }
  else{
	Grid = fff_matrix_fromPyArray( grid );
  }
  fff_vector *Density = fff_vector_new(Grid->size1);
  if (nsamplings>0)
	fff_BGMM_Gibbs_sampling(Density, BG, X,Grid, nsamplings,method);
  
  density = fff_vector_toPyArray( Density);
  
  if (grid !=NULL)
	fff_matrix_delete(Grid);

  fff_BGMM_delete( BG );

  fff_matrix_delete(X);

  fff_matrix_delete(PriorPrecision);
  fff_matrix_delete(PriorCenters);
  fff_vector_delete(PriorMeanScale);
  fff_vector_delete(PriorDof);
  fff_vector_delete(PriorWeights);

  mean = fff_matrix_toPyArray( Mean); 
  mean_scale = fff_vector_toPyArray( MeanScale); 
  membership = fff_matrix_toPyArray( Membership ); 
  precision_scale = fff_matrix_toPyArray( PrecisionScale );
  weights = fff_vector_toPyArray( Weights );
  dof = fff_vector_toPyArray( Dof );

  PyObject *ret = Py_BuildValue("NNNNNNN",membership, mean, mean_scale, precision_scale, weights,dof,density );
  return ret; 
}


static PyArrayObject* dpmm(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *precisions,*labels,*prior_means, *prior_mean_scale, *density;
  double alpha;
 
  int k,dim,niter = 10;
  int nis = 10;
  PyArrayObject *grid = NULL;
  double dof = 0;
  
  int OK = PyArg_ParseTuple( args, "O!dO!O!O!O!|iO!id:dpmm", 
							 &PyArray_Type, &x,
							 &alpha,
							 &PyArray_Type, &precisions,
							 &PyArray_Type, &prior_means,
							 &PyArray_Type, &prior_mean_scale,
							 &PyArray_Type, &labels, 
							 &niter,
							 &PyArray_Type, &grid,
							 &nis,
							 &dof
							 );
  if (!OK) return NULL; 
  
  fff_matrix *X = fff_matrix_fromPyArray( x ); 
  dim = X->size2;
 
  int type = 1; 
  if (dof==0) type = 0;
  fff_IMM *IMM =  fff_IMM_new( alpha, dim, type );
  
  fff_vector *Precisions = fff_vector_fromPyArray( precisions ); 
  fff_vector *Prior_means = fff_vector_fromPyArray( prior_means );
  fff_vector *Prior_mean_scale = fff_vector_fromPyArray( prior_mean_scale );
  fff_array *Labels = fff_array_fromPyArray( labels );
  
  if (type==0)
	fff_fixed_IMM_instantiate(IMM, Precisions, Prior_means, Prior_mean_scale);
  else
	fff_var_IMM_instantiate(IMM, Precisions, Prior_means, Prior_mean_scale,dof);
  fff_vector_delete(Precisions);
  fff_vector_delete(Prior_means);
  fff_vector_delete(Prior_mean_scale);
  
  fff_array *Z = fff_array_new1d(FFF_LONG,Labels->dimX);
  k = fff_IMM_estimation(IMM, Z, X, Labels, niter);

  fff_matrix* Grid;
	
  if (grid == NULL){
	Grid = fff_matrix_new(X->size1,X->size2);
	fff_matrix_memcpy(Grid,X); 
  }
  else{
	Grid = fff_matrix_fromPyArray( grid );
  }
  
  fff_vector* Density = fff_vector_new(Grid->size1);
  fff_IMM_sampling(Density, IMM, Z, X, Labels, Grid, nis);
  fff_matrix_delete(Grid); 
  density = fff_vector_toPyArray( Density );
  
  fff_array_delete(Labels);
  fff_array_delete(Z);
 
  
  fff_IMM_delete( IMM );
  fff_matrix_delete(X);
 
  return density;
}

static PyObject* fdp(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *precisions, *pvals, *labels, *density, *posterior;
  double alpha, g0,g1,dof;

  int k,dim,niter = 1000;
  int nis = 1000;
  int nii = 1000;
  PyArrayObject *grid = NULL;
  
  int OK = PyArg_ParseTuple( args, "O!ddddO!O!O!|iO!ii:fdp", 
							 &PyArray_Type, &x,
							 &alpha,
							 &g0,
							 &g1,
							 &dof,
							 &PyArray_Type, &precisions,
							 &PyArray_Type, &pvals,
							 &PyArray_Type, &labels,
							 &niter,
							 &PyArray_Type, &grid,
							 &nis,
							 &nii
							 ); 
  if (!OK) Py_RETURN_NONE; 
 
  fff_matrix *X = fff_matrix_fromPyArray( x );
  fff_matrix *Precisions = fff_matrix_fromPyArray( precisions ); 
  fff_vector *Pvals = fff_vector_fromPyArray( pvals );
  fff_array *Labels = fff_array_fromPyArray( labels );
  dim = X->size2;
  
  fff_FDP *FDP =  fff_FDP_new( alpha, g0, g1,dim,dof );
  fff_FDP_instantiate(FDP, Precisions);
  fff_matrix_delete(Precisions);

  fff_array *Z = fff_array_new1d(FFF_LONG,Labels->dimX);
  k = fff_FDP_estimation(FDP, Z, X, Pvals, Labels, niter);
  fff_matrix* Grid;
  if (grid == NULL){
	Grid = fff_matrix_new(X->size1,X->size2);
	fff_matrix_memcpy(Grid,X);
  }
  else
	Grid = fff_matrix_fromPyArray( grid );
  fff_vector* Density = fff_vector_new(Grid->size1);
  fff_FDP_sampling(Density, FDP, Z, X, Pvals, Labels, Grid, nis);
  fff_matrix_delete(Grid);

  fff_vector* Post = fff_vector_new(X->size1);
  fff_FDP_inference(FDP, Z, Post, X, Pvals, Labels, nii);

  fff_vector_delete(Pvals);
  fff_array_delete(Labels);
  fff_array_delete(Z);
 
  // get the results as python arrrays
  density = fff_vector_toPyArray( Density );
  posterior = fff_vector_toPyArray( Post );
  
  fff_FDP_delete( FDP );
  fff_matrix_delete(X);
	
  // Output tuple 
  PyObject *ret = Py_BuildValue("NN",density,posterior);
  return ret;
  //Py_RETURN_NONE;
}


static PyMethodDef module_methods[] = {
  {"ward",    /* name of func when called from Python */
   (PyCFunction)ward,      /* corresponding C function */
   METH_KEYWORDS,   /* ordinary (not keyword) arguments */
   ward_doc}, /* doc string */
    {"cmeans",    /* name of func when called from Python */
   (PyCFunction)cmeans,      /* corresponding C function */
   METH_KEYWORDS,   /* ordinary (not keyword) arguments */
   cmeans_doc}, /* doc string */
  {"voronoi",    /* name of func when called from Python */
   (PyCFunction)voronoi,      /* corresponding C function */
   METH_KEYWORDS,   /* ordinary (not keyword) arguments */
   voronoi_doc}, /* doc string */
  {"fcm",    /* name of func when called from Python */
   (PyCFunction)fcm,      /* corresponding C function */
   METH_KEYWORDS,   /* ordinary (not keyword) arguments */
   fcm_doc}, /* doc string */
  /*
  {"constrained_cmeans", 
   (PyCFunction)constrained_cmeans,
   METH_KEYWORDS,   
   constrained_cmeans_doc},
   {"constrained_match",   
   (PyCFunction)constrained_match,
   METH_KEYWORDS,  
   constrained_match_doc},
  */
  {"gmm", 
   (PyCFunction)gmm,
   METH_KEYWORDS,
   gmm_doc},
  {"gmm_relax",
   (PyCFunction)gmm_relax,
   METH_KEYWORDS,
   gmm_relax_doc},
  {"gmm_partition",
   (PyCFunction)gmm_partition,
   METH_KEYWORDS,
   gmm_partition_doc},
  {"gmm_membership",
   (PyCFunction)gmm_membership,
   METH_KEYWORDS,
   gmm_membership_doc},
  {"bayesian_gmm",
   (PyCFunction)bayesian_gmm,
   METH_KEYWORDS,
   bayesian_gmm_doc},
  {"bayesian_gmm_sampling",
   (PyCFunction)bayesian_gmm_sampling,
   METH_KEYWORDS,
   bayesian_gmm_sampling_doc},
  {"gibbs_gmm",
   (PyCFunction)gibbs_gmm,
   METH_KEYWORDS,
   gibbs_gmm_doc},
  {"gmm_shift",
   (PyCFunction)gmm_shift,
   METH_KEYWORDS,
   gmm_shift_doc},
  {"fdp",
   (PyCFunction)fdp,
   METH_KEYWORDS,
   fdp_doc},
  {"dpmm",
   (PyCFunction)dpmm,
   METH_KEYWORDS,
   dpmm_doc}, 
  /*
	{"hgmm",  
	(PyCFunction)hgmm,
	METH_KEYWORDS,  
	hgmm_doc},
  */
  {NULL, NULL,0,NULL}
};


void init_clustering(void)
{
  Py_InitModule3("_clustering", module_methods, module_doc);
  fffpy_import_array();
  import_array();   /* required NumPy initialization */
}
