/*! 
  \file fff_glm_twolevel.h
  \brief General linear model under observation errors (mixed effects) 
  \author Alexis Roche
  \date 2008
  
  Bla bla bla

  */


#ifndef FFF_GLM_TWOLEVEL
#define FFF_GLM_TWOLEVEL

#ifdef __cplusplus
extern "C" {
#endif

#include "fff_vector.h"
#include "fff_matrix.h"
  

  /*! 
    \struct fff_glm_twolevel_EM 
    \brief Structure for the mixed-effect general linear model 

    This structure is intended for multiple regression under mixed
    effects using the EM algorithm.  
  */
  typedef struct{
    
    size_t n; /*! Number of observations */ 
    size_t p; /*! Number of regresssors */ 
    fff_vector* b; /*! Effect estimate */
    double s2; /*! Variance estimate */
    fff_vector* z; /*! Expected true effects */
    fff_vector* vz; /*! Expected variance of the true effects (diagonal matrix) */
    fff_vector* Qz; /* Expected prediction error */ 
    unsigned int niter; /* Number of iterations */ 

  } fff_glm_twolevel_EM; 


  extern fff_glm_twolevel_EM* fff_glm_twolevel_EM_new(size_t n, size_t p); 

  extern void fff_glm_twolevel_EM_delete(fff_glm_twolevel_EM* thisone); 
  extern void fff_glm_twolevel_EM_init(fff_glm_twolevel_EM* em); 
  /*

  \a PpiX is defined by: \f$ PpiX = P (X'X)^{-1} X' \f$, where: \f$ P
  = I_p - A C (C' A C)^{-1} C' \f$ with \f$ A = (X'X)^-1 \f$ is the
  appropriate projector onto the constaint space, \f$ C'b=0 \f$. \a P
  is, in fact, orthogonal for the dot product defined by \a X'X.
  
  Please note that the equality \a PpiX*X=P should hold but is not
  checked.

  */
  extern void fff_glm_twolevel_EM_run(fff_glm_twolevel_EM* em, const fff_vector* y, const fff_vector* vy, 
				 const fff_matrix* X, const fff_matrix* PpiX, unsigned int niter); 

  extern double fff_glm_twolevel_log_likelihood( const fff_vector* y, 
						 const fff_vector* vy, 
						 const fff_matrix* X, 
						 const fff_vector* b, 
						 double s2, 
						 fff_vector* tmp );

#ifdef __cplusplus
}
#endif

#endif
