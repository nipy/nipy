/*!
  \file fff_GMM.h
  \brief Gaussian Mixture Models
  \author Bertrand Thirion
  \date 2004-2006

  This library implements different kinds of GMM estimation techniques

  

  */

#ifndef fff_GMM
#define fff_GMM

#ifdef __cplusplus
extern "C" {
#endif

  
#include "fff_graphlib.h"

  typedef struct fff_GMM_{
	
    long k;            /* number of components */
    long dim;          /* feature dimension */
	int prec_type;     /* type of teh precision */
	fff_matrix * means;
	fff_matrix * precisions;
	fff_vector * weights;
    
  } fff_GMM_;



  typedef struct fff_Bayesian_GMM{
	
    long k;            /* number of components */
    long dim;          /* feature dimension */
	
	/* it is assumed that 
	   1) the covariance is diagonal, so that 
	   precision matrices and centroid matrices do have the same size:
	   (k, dim), 
	   where k is the number of components in the mixture
	   and dim is the number of dimensions of the data
	   All the matrices in the model have this size.
	   All the vectors have length k.
	   2)Conjugate normal-Wishart prior are used 
	*/

	/* priors */
	fff_matrix * prior_means;
	fff_vector * prior_means_scale;
	fff_matrix * prior_precisions;
	fff_vector * prior_dof;
	fff_vector * prior_weights;

    /* current estimates */
	fff_vector * means_scale;
	fff_matrix * means;
	fff_matrix * precisions;
	fff_vector * dof;
	fff_vector * weights;
    
  } fff_Bayesian_GMM;

  /*! 
    \brief Constructor for the GMM structure 
    \param k : number of components
    \param dim : the number of dimensions
	\param prec_type: (int) the parameterization of the model
	
	if prec_type==0 the model is assumed to be Gaussian,
	heteroscedesatic with full covariance.
	if prec_type==1 the model is assumed to be Gaussian,
	heteroscedesatic and diagonal.
	if prec_type==2 the model is assumed to be Gaussian,
	homoscedastic, diagonal and with equal weights.
  */
  extern fff_GMM_* fff_GMM_new( const long k, const long dim,const int prec_type );
  
  /*! 
    \brief Destructor for the GMM structure
    \param thisone the GMM to be deleted
  */
  extern int fff_GMM_delete( fff_GMM_* thisone );
  
     /*!
    \brief Labelling checking algorithm 
    \param Label vector
    \param k number of classes

    This algorithm quickly checks that for each i in [0,..,k-1]
    there exists an integer j so that Label[j] = i.
    it returns 1 if yes, 0 if no
  */
  extern int fff_clustering_OntoLabel(const fff_array * Label, const long k);

  /*!
    \brief GMM algorithm with selection of the number of clusters
    \param X data matrix 
    \param Centers Cluster centers
    \param Precision Cluster precisions
    \param Weights Mixture Weights
    \param Label data labels
    \param nbclust number of clusters for which a mixture is searched. 
    \param maxiter maximum number of iterations
    \param delta small constant for control of convergence

    This function performs different initializations of the GMM clustering
    algorithm with nbclust mixtures, and keeps the best one according
    to a BIC criterion. 

    The optimal number of clusters is returned.  It is assumed that Centers,
    Precision, and Weights have been allocated for max(nbclust)
    clusters.
  */
  int fff_clustering_gmm_select( fff_matrix* Centers, fff_matrix* Precision,  fff_vector *Weights, fff_array *Label, const fff_matrix* X, const fff_vector *nbclust, const int maxiter, const double delta);
  /*!
    \brief GMM algorithm
    \param X data matrix 
    \param Centers Cluster centers
    \param Precision Cluster precisions
    \param Weights Mixture Weights
    \param Label data labels
    \param maxiter maximum number of iterations
    \param delta small constant for control of convergence
    \param ninit number of initializations

    This function performs ninit initializations of the GMM clustering
    algorithm, and keeps the best one. The average log-likelihood is
    returned.
    
  */
  extern double fff_clustering_gmm_ninit( fff_matrix* Centers, fff_matrix* Precision,  fff_vector *Weights, fff_array *Label, const fff_matrix* X, const int maxiter, const double delta, const int ninit );
 /*!
    \brief GMM algorithm
    \param X data matrix 
    \param Centers Cluster centers
    \param Precision Cluster precisions
    \param Weights Mixture Weights
    \param Label data labels
    \param maxiter maximum number of iterations
    \param delta small constant for control of convergence
    \param chunksize the number of features on which gmm is performed
	\param verbose verbosity mode

    This algorithm performs a GMM of the data X.  Note that the data
    and cluster matrices should be dimensioned as (nb items * feature
    dimension) and (nb clusters * feature dimension).
    The precision is written either in the form 
    - (nb clusters * sqf), where sqf is the square of the feature dimension.  
    - (nb clusters * feature dimension), then it is assumed that it codes for a
    diagonal precision matrix, and an algorithm with diagonal
    covariance is implemented. 
    - (1* sqf) then the Gaussians are assumed to be homoscedastic. Only
    one diagonal variance is computed  for all clusters. the weights of 
    the mixture are set (1/nb clusters) 
    
    chunksize has to be chosen in the interval [nb clusters, nb features]
    chunksize = nb features is a standard choice, 
    but lowering chunksize will profit to  computation time.
    it is advisable to have  chunksize<10^6
    
    The metric used in the algorithm is Euclidian.
    The returned Label vector is a hard membership function 
    computed after convergence.
    The normlized log-likelihood is returned.
  */
  extern double fff_clustering_gmm( fff_matrix* Centers, fff_matrix* Precision,  fff_vector *Weights, fff_array *Label, const fff_matrix* X, const int maxiter, const double delta, const int chunksize, const int verbose );
 

  /*!
    \brief Evaluation of a GMM algorithm on a dataset
    \param X data matrix 
    \param Centers Cluster centers
    \param Precision Cluster precisions
    \param Weights Mixture Weights
    \param L average log-likelihood

    This algorithm computes the average log-likelihood of a GMM on an
    empirical dataset. This number is returned.
    The algorithm adapts to the size of the precision matrix
    - (k*(f*f)) precision is a full f*f matrix for all clusters
    - (k*f) precision diagonal
    - (1*f) precision diagonal and equal for all clusters
  */
  extern double fff_gmm_mean_eval(double* L, const fff_matrix* X, const fff_matrix* Centers, const fff_matrix* Precision, const fff_vector* Weights);

/*!
    \brief Relaxation of a previously estimated GMM on a dataset
    \param X data matrix 
    \param Centers Cluster centers
    \param Precision Cluster precisions
    \param Weights Mixture Weights
    \param LogLike  log-likelihood of each data
	\param Labels final labelling

    This algorithm computes the average log-likelihood of a GMM on an
    empirical dataset. This number is returned.
    The algorithm adapts to the size of the precision matrix
    - (k*(f*f)) precision is a full f*f matrix for all clusters
    - (k*f) precision diagonal
    - (1*f) precision diagonal and equal for all clusters
  */
  extern int fff_gmm_relax( fff_vector* LogLike, fff_array* Labels, fff_matrix* Centers, fff_matrix* Precision, fff_vector* Weights, const fff_matrix* X, const int maxiter, const double delta);

  /*!
    \brief Evaluation of a GMM on a dataset and labelling of the dataset X
    \param X data matrix 
    \param Centers Cluster centers
    \param Precision Cluster precisions
    \param Weights Mixture Weights
    \param LogLike  log-likelihood of each data
    \param Labels final labelling

    This algorithm computes the average log-likelihood of a GMM on an
    empirical dataset. This number is returned.
    The algorithm adapts to the size of the precision matrix
    - (k*(f*f)) precision is a full f*f matrix for all clusters
    - (k*f) precision diagonal
    - (1*f) precision diagonal and equal for all clusters
  */
  extern int fff_gmm_partition(fff_vector* LogLike, fff_array* Labels, const fff_matrix* X, const fff_matrix* Centers, const fff_matrix* Precision, const fff_vector* Weights);

    /*!
    \brief representation of GMM membership with a graph structure
	\param G Weighted Graph
	\param X data-defining matrix 
    \param Centers Cluster centers
    \param Precision Cluster precisions
    \param Weights Mixture Weights

    This algorithm computes the probabilistic membership values 
	of the data X with respect to the GMM model and stores this 
	-presumably sparse- information in a graph structure.
	If G->E = 0, nothing is done with graph
	Otherwise, the  memebership is coded in graph
	The number of edges is retuned.
	In practice, it is thus advised to call it once with an empty graph
	to get the number of edges, then 
	to allocate the graph structure and recall the function.
  */
  extern int fff_gmm_membership(fff_graph* G, const fff_matrix* X, const fff_matrix* Centers, const fff_matrix* Precision, const fff_vector* Weights);


   /*!
    \brief Shifting the points in X toward points of larger likelihood
	\param X moving points
    \param Centers Cluster centers
    \param Precision Cluster precisions
    \param Weights Mixture Weights

  */
  extern int fff_gmm_shift(fff_matrix* X, const fff_matrix* Centers, const fff_matrix* Precision, const fff_vector* Weights);


/*!
    \brief Implementation of a Byaesian GMM model based on the Variational Bayes Model.
    \param BGMM  the structure that contains the Bayesian GMM
	\param Label final data labelling
	\param X data matrix 
	\param maxiter Maximal number of iterations
    \param delta minimal relative increment of the likelihood to declare convergence
	The Model comes from Penny, 2001 (research report).

	Given a dataset X, and prior about clsuter centers and precision
    the dunction computes a GMM model (Centers,Precision,Weights)

*/
  extern double fff_VB_gmm_estimate(fff_Bayesian_GMM* BGMM,  fff_array *Label, const fff_matrix* X, const int maxiter, const double delta);



   /*! 
    \brief Constructor for the Bayesian GMM structure 
    \param k : number of components
    \param dim : the number of dimensions
  */
  extern fff_Bayesian_GMM* fff_BGMM_new( const long k, const long dim );
  
  /*! 
    \brief Destructor for the Bayesian GMM structure
    \param thisone the BGMM to be deleted
  */
  extern int fff_BGMM_delete( fff_Bayesian_GMM* thisone );
  
  /*
	\brief instantiation of the prior of the Bayesian GMM
	\param BG the Bayesian GMM to be instantiated
	\param prior_means the prior of the means
	\param prior_means_scale scaling factor on the precision of the means
	\param prior_precision prior precision in the wishart model
	\param prior_dof number of degrees of freedom in the Wishart model
	\param prior_weight prior on the weights in the Dirichlet model
	
	Note that all the parameters are copied
   */
  extern int fff_BGMM_set_priors(fff_Bayesian_GMM* BG, const fff_matrix * prior_means, const fff_vector *prior_means_scale, const fff_matrix * prior_precision, const fff_vector* prior_dof, const fff_vector *prior_weights );

   /*
	\brief Estimation of the BGMM using a Gibbs sampling technique
	\param membership the average membership variable across iterations
	\param BG the BGMM to be estimated
	\param X the data used in the estimation of the model
	\param niter the number of iterations in the sampling
	\param method choice of a quick(purely normal) technique

	This function performs the estimation of the model.
	The membership (hidden) variable is randomly resampled at each step,
	while the parameters of the model are recomputed analytically
	for the sake of speed.
	The number of itertaions is niter for bith the burun-in 
	and the estimation periods
	The average data memebership across iterations is given in membership.
	The BGMM is re-instantiated with the average parameters 
	during the estimation period.
   */

  extern int fff_BGMM_Gibbs_estimation(fff_matrix* membership, fff_Bayesian_GMM* BG, const fff_matrix *X, const int niter, const int method);

    
   /*
	\brief Sampling of the BGMM using on a grid 
	\param the average density on the given grid 
	\param BG the BGMM to be estimated
	\param X the data used in the estimation of the model
	\param grid the grid used in the estimation of the model
	\param niter the number of iterations in the sampling
	\param method choice of a quick(purely normal) technique

   */
  extern int fff_BGMM_Gibbs_sampling(fff_vector* density, fff_Bayesian_GMM* BG, const fff_matrix *X, const fff_matrix *grid, const int niter, const int method);


  /*
	\brief Reading out the BGMM structure
	\param means current estimate of the cluster centers
	\param means_scale  current precision on the means matrices
	\param precisions current precision estimate
	\param dof current dif of the Wisharet model
	\param weights current weights of the components 
	\param BG BGMM model.

	Note that all the matrices are copied.
   */
  extern int fff_BGMM_get_model( fff_matrix * means, fff_vector * means_scale, fff_matrix * precisions, fff_vector* dof, fff_vector * weights, const fff_Bayesian_GMM* BG);

  /*
	\brief Writing the BGMM structure	
	\param BG BGMM model.
	\param means current estimate of the cluster centers
	\param means_scale  current precision on the means matrices
	\param precisions current precision estimate
	\param dof current dif of the Wisharet model
	\param weights current weights of the components 

	Note that all the matrices are copied.
   */
  extern int fff_BGMM_set_model( fff_Bayesian_GMM* BG, const fff_matrix * means, const fff_vector * means_scale, const fff_matrix * precisions, const fff_vector* dof, const fff_vector * weights );


  /*!	
	\brief Sampling of a Bayesian GMM model based on the Variational Bayes.
	\param density the posterior
    \param BGMM  the structure that contains the Bayesian GMM
	\param grid the smapling grid for the posterior

*/  
  extern int fff_BGMM_sampling(fff_vector * density, fff_Bayesian_GMM* BG, const fff_matrix *grid);

/*!	
	\brief Sampling of a Bayesian GMM model 
	\param density the posterior as a (nbitem,BG->k) matrix, where nbitem X->size1
	\param X the smapling grid for the posterior
	\param BGMM  the structure that contains the Bayesian GMM

*/  
  extern double fff_WNpval(fff_matrix * proba, const fff_matrix *X, const fff_Bayesian_GMM* BG);

#ifdef __cplusplus
}
#endif

#endif
