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
/*
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
*/

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


static char module_doc[] =
" Clustering routines.\n\
Author: Bertrand Thirion (INRIA Saclay, Orsay, France), 2004-2008.";

/*
static PyObject* ward(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *cost, *parent ;
  int n,q;

  int OK = PyArg_ParseTuple( args, "O!:ward", 
							 &PyArray_Type, &x);
	
    if (!OK) Py_RETURN_NONE;
  
	fff_matrix* X = fff_matrix_fromPyArray( x );
	n = X->size1;
	q = 2*n-1;
	
  	
	fff_array *Parent = fff_array_new1d(FFF_LONG,q);
	fff_vector *Cost = fff_vector_new(q);
	fff_vector_set_all(Cost,0);
	
	fff_clustering_ward(Parent,Cost,X);
	
	cost = fff_vector_toPyArray( Cost ); 
	parent = fff_array_toPyArray( Parent );
	fff_matrix_delete(X);
	
	PyObject *ret = Py_BuildValue("NN",parent,cost);
	return ret;
}
*/


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
 
  /* get the results as python arrrays */
  density = fff_vector_toPyArray( Density );
  posterior = fff_vector_toPyArray( Post );
  
  fff_FDP_delete( FDP );
  fff_matrix_delete(X);
	
  /* Output tuple */
  PyObject *ret = Py_BuildValue("NN",density,posterior);
  return ret;
}

static PyObject* fdp2(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *precisions, *pvals, *labels, *co_clust, *posterior, *density;
  double alpha, g0,g1,dof;

  int k,dim,niter = 1000;
  int nii = 1000;
  int nis = 1000;
  PyArrayObject *grid = NULL;
  
  int OK = PyArg_ParseTuple( args, "O!ddddO!O!O!|iO!ii:fdp2", 
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
  if (!OK) {
    printf("argument error in fdp2\n");
    Py_RETURN_NONE; 
  }
 
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

  fff_matrix *CoCluster = fff_matrix_new(X->size1,X->size1);
  fff_vector* Post = fff_vector_new(X->size1);
  fff_FDP_inference2(FDP, Z, Post, CoCluster, X, Pvals, Labels, nii);
  
  fff_vector* Density = fff_vector_new(Grid->size1);
  fff_FDP_sampling(Density, FDP, Z, X, Pvals, Labels, Grid, nis);
  fff_matrix_delete(Grid);
  
  fff_vector_delete(Pvals);
  fff_array_delete(Labels);
  fff_array_delete(Z);
 
  /* get the results as python arrrays */
  co_clust = fff_matrix_toPyArray( CoCluster );
  posterior = fff_vector_toPyArray( Post );
  density = fff_vector_toPyArray( Density );

  fff_FDP_delete( FDP );
  fff_matrix_delete(X);
	
  /* Output tuple */
  PyObject *ret = Py_BuildValue("NNN", co_clust, posterior, density);
  return ret;
}

static PyMethodDef module_methods[] = {
  /*
  {"ward",    
   (PyCFunction)ward,
   METH_KEYWORDS,   
   ward_doc},
  */
  {"fdp",
   (PyCFunction)fdp,
   METH_KEYWORDS,
   fdp_doc},
  {"fdp2",
   (PyCFunction)fdp2,
   METH_KEYWORDS,
   fdp_doc},
  {NULL, NULL,0,NULL}
};


void init_clustering(void)
{
  Py_InitModule3("_clustering", module_methods, module_doc);
  fffpy_import_array();
  import_array();   /* required NumPy initialization */
}
