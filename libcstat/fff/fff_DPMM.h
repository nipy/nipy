/*!
  \file fff_DPMM.h
  \brief Dirichlet Process Mixture Models
  \author Bertrand Thirion
  \date 2007

  This library implements different kinds of DPMM estimation techniques

*/

#ifndef fff_DPMM
#define fff_DPMM

#ifdef __cplusplus
extern "C" {
#endif

#include "fff_array.h"
#include "fff_vector.h"
#include "fff_matrix.h"

  
  /*************************************************************/
  /**************** IMM ****************************************/
  /*************************************************************/

/* 
   This is a Dirichlet Process (Gaussian) Mixture Model structure:
   in fact this is a "classical" DPMM with Gaussian components and
   fixed diagonal covariance.  Another way to see that is to
   consider it as a virtually infinite GMM (with fixed diagonal
   covariance) where the classes are created when needed.
*/
  
  typedef struct fff_IMM{
	double alpha;
	long dim;
	long k;	
	int type;
	/**/

	fff_matrix * means;
	/* this is the (component,dimension) mean of the centers*/ 
	fff_vector * weights;
	/* this is the (component) weight vector */ 
	fff_matrix * prec_means;
	/* this is the (component,dimension) precision on the centers*/ 

	fff_vector * prior_precisions;
	/* this is the (dimension) precision on the components*/ 
	fff_vector * prior_means;
	/* this is the (dimension) prior on the centers*/ 

	fff_vector * prior_mean_scale;
	/* this is the (dimension) prior precision on the centers*/ 

	fff_matrix *empmeans;
	/* this is the (component,dimension) empirical value of the centers*/ 
	fff_array *pop;
	/* this is the (component_wise precision */ 
	
	fff_matrix * precisions;
	/* this is the (component,dimension) prior precision on the components*/ 
	double prior_dof;
	fff_vector * dof;


  }fff_IMM;
  
  /*! 
    \brief Constructor for the DPMM structure 
	\param alpha alpha parameter
	\param dim dimension of the MM  
  */
  extern fff_IMM* fff_IMM_new( const double alpha, const long dim, const int type);
  
  /*! 
    \brief Destructor for the DPMM structure
    \param thisone the DPMM to be deleted
  */
  extern int fff_IMM_delete( fff_IMM* thisone );

   /*! 
    \brief instantiation of the IMM struct with fixed precision
    \param IMM the previously created IMM
	\param prior_precision the precisions of the process
	\param prior_means the prior on the means
	\param prior_mean_scale the prior precision one the means

	Note that the precisions remain fixed during the estimation
	procedure.It is also assumed that the precisions are diagonal:
	precisions should thus be a (dim) fff_vector.
	prior_means and prior_mean_scale are vector of length dim

  */
  extern int fff_fixed_IMM_instantiate(fff_IMM* IMM, const fff_vector * prior_precisions, const fff_vector* prior_means, const fff_vector* prior_mean_scale);
   
  /*! 
    \brief instantiation of the IMM struct with variable precision
    \param IMM the previously created IMM
	\param prior_precision the precisions of the process
	\param prior_means the prior on the means
	\param prior_mean_scale the prior precision one the means
	\param prior_dof the prior dof

	Note that the precisions remain fixed during the estimation
	procedure.It is also assumed that the precisions are diagonal:
	precisions should thus be a (dim) fff_vector.
	prior_means and prior_mean_scale are vector of length dim.
	prior_dof is used in the variable covariance scheme only, ie if 
	type==1.

  */
  extern int fff_var_IMM_instantiate(fff_IMM* IMM, const fff_vector * prior_precisions, const fff_vector* prior_means, const fff_vector* prior_mean_scale, const double prior_dof);

  /*! 
    \brief Estimator for the IMM structure
	\param IMM the IMM to be estimated
    \param Z the membership vector 
	\param data the data on which the estimation is based
	\param labels chunking model (see below)
	\niter number of iterations in the estimation

	- data should ahabe size (nbitems, dim) 
	- Z, labels, pvals should have length nbitems

	Z is useful only when estimation is used as burn-in period: it
	returns the state of the membership at the end of the burn-in
	period, so that samples can then be drawn form the true posterior.
	
	labels are used to make chunks of data in the posterior
	computation; this can be either (0:nbitems-1) or contain more
	complex information

	The estimation is a straightforward implementation of Neal'98 paper.
	with fixed caviariance
  */
  extern int fff_IMM_estimation(fff_IMM* IMM, fff_array *Z, const fff_matrix *data, const fff_array * labels, const long niter);

  
/*! 
    \brief Sampling from the IMM structure
	\param desnity/likelihood of the data sampled on grid
   	\param IMM the IMM to be estimated
    \param Z the membership vector 
	\param data the data on which the estimation is based
	\param labels chunking model (see below)
	\param grid grid where the data is sampled
	\param niter number of iterations in the estimation

	idem fff_IMM estimation, but an addition grid is provided, so that the likelihood of the data is sampled on that grid.
	grid should have size (nbnodes, dim), 
	density should have size (nbnodes)
	
	k = current number of components is returned
  */
  extern int fff_IMM_sampling(fff_vector *density, fff_IMM* IMM, fff_array *Z, const fff_matrix *data, const fff_array * labels, const fff_matrix *grid, const long niter);


  /*
	\brief Getting the current parameters of the model
	\param mean center matrix
	\param prec_means precision on the mean matrix
	\param weights weights vector
	\param IMM the IMM from which parameters are extracted

	if the IMM currently has  k>1 components, 
	mean and precision should have size (k-1,dim)
	weights should have size (k)
*/
	extern int fff_IMM_get_model( fff_matrix * mean, fff_matrix * prec_means, fff_vector * weights, const fff_IMM* IMM);
 


  /*************************************************************/
  /**************** FDP ****************************************/
  /*************************************************************/


  /* 
	   This is a customized IMM:
	   in fact this is a "classical" DPMM with Gaussian components
	   and fixed diagonal covariance, but 
	   1. an additional null class is added with a uniform density
	   and ech item is assumed to have a certain probability 
	   of belonging to the null class
	   2. The base measure is not gaussian but uniform, i.e. 
	   equal to some constant, which is provided bu the user
	   A description of this model can be found in IPMI'O7, Thirion et al.
	*/
  typedef struct fff_FDP{
	
	double g0;
	double g1;
	double alpha;
	long dim;
	long k;
	double prior_dof;
	
	fff_matrix * means;
	fff_matrix * precisions;
	fff_vector * weights;

    /* fff_matrix *empmeans; */
	fff_array *pop;
	fff_matrix * prior_precisions;
	
  }fff_FDP;



  
  /*! 
    \brief Constructor for the FDP structure 
	\param alpha alpha parameter
	\param g0 constant of the uniform distribution ; used for the null hypothesis
	\param g1 the same as g0, but for the alternative hypothesis
	\param dim dimension of the MM
  
  */
  extern fff_FDP* fff_FDP_new( const double alpha, const double g0, const double g1, const long dim, const double prior_dof);
	
  /*! 
    \brief Destructor for the FDP structure
    \param thisone the FDP to be deleted
  */
	extern int fff_FDP_delete( fff_FDP* thisone );

   /*! 
    \brief instantiation of the FDP struct
    \param FDP the previously created FDP
	\param precision the precisions of the process

	Note that the precisions remain fixed during the estimation
	procedure.It is also assumed that the precisions are diagonal:
	precisions should thus be a (1,dim) fff_matrix.
  */
  extern int fff_FDP_instantiate(fff_FDP* FDP, const fff_matrix * precisions);

   /*! 
    \brief Estimator for the FDP structure
	\param FDP the FDP to be estimated
    \param Z the membership vector 
	\param data the data on which the estimation is based
	\param pvals the prior probability of H1 given the data
	\param labels chunking model (see below)
	\niter number of iterations in the estimation

	- data should ahabe size (nbitems, dim) 
	- Z, labels, pvals should have length nbitems

	Z is useful only when estimation is used as burn-in period: it
	returns the state of the membership at the end of the burn-in
	period, so that samples can then be drawn form the true posterior.
	
	labels are used to make chunks of data in the posterior
	computation; this can be either (0:nbitems-1) or contain more
	complex information
  */
  extern int fff_FDP_estimation(fff_FDP* FDP, fff_array *Z, const fff_matrix *data, const fff_vector * pvals, const fff_array * labels, const long niter);

  /*! 
    \brief Inference based on the FDP structure
	\param FDP the FDP to be estimated
    \param Z the membership vector 
	\param posterior the posterior probability of H1 given the data
	\param data the data on which the estimation is based
	\param pvals the prior probability of H1 given the data
	\param labels chunking model (see below)
	\param niter number of iterations in the estimation

	idem fff_FDP estimation, but the posterior proba of H1 given the data is returned. In that case, Z should be corrected instantiated so that the MC is in its steady state.
  */ 
  extern int fff_FDP_inference(fff_FDP* FDP, fff_array *Z, fff_vector* posterior, const fff_matrix *data, const fff_vector * pvals, const fff_array * labels, const long niter);
  
/*! 
    \brief Sampling from the FDP structure
	\param desnity likelihood of the data under H1 sampled on grid
   	\param FDP the FDP to be estimated
    \param Z the membership vector 
	\param data the data on which the estimation is based
	\param pvals the prior probability of H1 given the data
	\param labels chunking model (see below)
	\param grid grid where the data is sampled
	\param niter number of iterations in the estimation

	idem fff_FDP estimation, but an addition grid is provided, so that the likelihood of the data under H1 is sampled on that grid.
	grid should have size (nbnodes, dim), 
	density should have size (nbnodes)
	
	k = number of components is returned
  */
  extern int fff_FDP_sampling(fff_vector *density, fff_FDP* FDP, fff_array *Z, const fff_matrix *data, const fff_vector * pvals, const fff_array * labels, const fff_matrix *grid, const long niter);


  /*
	\brief Getting the current parameters of the model
	\param mean center matrix
	\param precision precision matrix
	\param weights weights vector
	\param FDP the FDP from which parameters are extracted

	if the FDP currently has  k>2 components, 
	mean and precision should have size (k-2,dim)
	weights should have size (k-1)
*/
	extern int fff_FDP_get_model( fff_matrix * mean, fff_matrix * precision, fff_vector * weights, const fff_FDP* FDP);


#ifdef __cplusplus
  }
#endif


#endif
