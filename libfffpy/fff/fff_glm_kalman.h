/*! 
  \file fff_glm_kalman.h
  \brief General linear model fitting using Kalman filters 
  \author Alexis Roche
  \date 2004-2006

  This library implements several Kalman filter variants to fit a
  signal (represented as a gsl_vector structure) in terms of a general
  linear model. Kalman filtering works incrementally as opposed to
  more classical GLM fitting procedures, hence making it possible to
  produce parameter estimates on each time frame. Two methods are
  currently available:
  
  - the standard Kalman filter: performs an ordinary least-square
  regression, hence ignoring the temporal autocorrelation of the
  errors.
  
  - the refined Kalman filter: original Kalman extension to estimate
  both the GLM parameters and the noise autocorrelation based on an
  autoregressive AR(1) model. Significantly more memory demanding than
  the standard KF.

  */


#ifndef FFF_GLM_KALMAN
#define FFF_GLM_KALMAN

#ifdef __cplusplus
extern "C" {
#endif

#include "fff_vector.h"
#include "fff_matrix.h"

#define FFF_GLM_KALMAN_INIT_VAR 1e7


  /*! 
    \struct fff_glm_KF
    \brief Standard Kalman filter structure.
    
  */
  typedef struct{

    size_t t;                /*!< time counter */
    size_t dim;              /*!< model dimension (i.e. number of linear regressors) */
    fff_vector* b;           /*!< effect vector */
    fff_matrix* Vb;          /*!< effect variance matrix before multiplication by scale */
    fff_vector* Cby;         /*!< covariance between the effect and the data before multiplication by scale */
    double ssd;              /*!< sum of squared residuals */
    double s2;               /*!< scale parameter (squared) */
    double dof;              /*!< degrees of freedom */
    double s2_cor;           /*!< s2 corrected for degrees of freedom, s2_cor=n*s2/dof */

  } fff_glm_KF;


  /*! 
    \struct fff_glm_RKF
    \brief Refined Kalman filter structure.
    
  */

  typedef struct{

    size_t t;               /*!< time counter */
    size_t dim;             /*!< model dimension (i.e. number of linear regressors) */
    fff_glm_KF* Kfilt;      /*!< standard kalman filter */
    fff_vector* db;         /*!< auxiliary vector for estimate variation */
    fff_matrix* Hssd;       /*!< SSD hessian (SSD = sum of squared differences) */
    double spp;             /*!< SSP value (SPP = sum of paired products) */
    fff_vector* Gspp;       /*!< SSP gradient */
    fff_matrix* Hspp;       /*!< SSP hessian */
    fff_vector* b;          /*!< effect vector */
    fff_matrix* Vb;         /*!< effect variance matrix before multiplication by scale */
    double s2;              /*!< scale parameter (squared) */
    double a;               /*!< autocorrelation parameter */
    double dof;             /*!< degrees of freedom */
    double s2_cor;          /*!< s2 corrected for degrees of freedom, s2_cor=n*s2/dof */
    fff_vector* vaux;       /*!< auxiliary vector */
    fff_matrix* Maux;       /*!< auxiliary matrix */

  } fff_glm_RKF;
  
  
  /*! \brief Constructor for the fff_glm_KF structure 
      \param dim model dimension (number of linear regressors)
  */
  extern fff_glm_KF* fff_glm_KF_new( size_t dim );
  /*! \brief Destructor for the fff_glm_KF structure
      \param thisone the fff_glm_KF structure to be deleted
  */
  extern void fff_glm_KF_delete( fff_glm_KF* thisone );
  /*! \brief Reset function (without destruction) for the fff_glm_KF structure 
      \param thisone the fff_glm_KF structure to be reset
  */
  extern void fff_glm_KF_reset( fff_glm_KF* thisone );
  /*! \brief Performs a standard Kalman iteration from a fff_glm_KF structure 
      \param thisone the fff_glm_KF structure to be iterated
      \param y current signal sample
      \param x current regressor values
  */
  extern void fff_glm_KF_iterate( fff_glm_KF* thisone, double y, const fff_vector* x );
  /*! \brief Constructor for the fff_glm_RKF structure 
      \param dim model dimension (number of linear regressors)
  */
  extern fff_glm_RKF* fff_glm_RKF_new( size_t dim );  
  /*! \brief Destructor for the fff_glm_RKF structure 
      \param thisone the fff_glm_KF structure to be deleted
   */
  extern void fff_glm_RKF_delete( fff_glm_RKF* thisone ); 
  /*! \brief Reset function (without destruction) for the fff_glm_RKF structure 
      \param thisone the fff_glm_KF structure to be reset
   */
  extern void fff_glm_RKF_reset( fff_glm_RKF* thisone );
  /*! \brief Performs a refined Kalman iteration from a fff_glm_RKF structure 
      \param thisone the fff_glm_KF structure to be iterated
      \param nloop number of refinement iterations 
      \param y current signal sample
      \param x current regressor values
      \param yy previous signal sample
      \param xx previous regressor values
   */
  extern void fff_glm_RKF_iterate( fff_glm_RKF* thisone, unsigned int nloop, 
				   double y, const fff_vector* x, 
				   double yy, const fff_vector* xx );
  /*!  
    \brief Perform an ordinary least square regression using the
  standard Kalman filter and return the degrees of freedom 
    \param thisone the fff_glm_KF structure to be filled in
    \param y input data
    \param X design matrix (column-wise stored covariates)
  */
  extern void fff_glm_KF_fit( fff_glm_KF* thisone,
			      const fff_vector* y,
			      const fff_matrix* X );
  
  /*!  
    \brief Perform a linear regression using the refined Kalman
    filter, corresponding to a GLM with AR(1) errors. 
    \param thisone the fff_glm_RKF structure to be filled in
    \param nloop number of refinement iterations 
    \param y input data
    \param X design matrix (column-wise stored covariates)
  */
  extern void fff_glm_RKF_fit( fff_glm_RKF* thisone,
			       unsigned int nloop, 
			       const fff_vector* y,
			       const fff_matrix* X );



#ifdef __cplusplus
}
#endif

#endif
