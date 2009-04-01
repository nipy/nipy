/*!
  \file fff_gen_stats.h
  \brief General interest statistical routines
  \author Alexis Roche
  \date 2004-2008

*/


 
#ifndef FFF_GEN_STATS
#define FFF_GEN_STATS

#ifdef __cplusplus
extern "C" {
#endif
  
#include "fff_vector.h"
#include "fff_matrix.h"

  
  /*!
    \brief Squared Mahalanobis distance
    \param x input data vector (beware: gets modified)
    \param S associated variance matrix
    \param Saux auxiliary matrix, same size as \a S
    
    Compute the squared Mahalanobis distance \f$ d^2 = x^t S^{-1} x
    \f$. The routine uses the Cholesky decomposition: \f$ S = L L^t
    \f$ where \a L is lower triangular, and then exploits the fact
    that \f$ d^2 = \| L^{-1}x \|^2 \f$.
  */  
  extern double fff_mahalanobis( fff_vector* x, fff_matrix* S, fff_matrix* Saux );

  /*
	\brief Generate a permutation from \a [0..n-1]
	\param x output list of integers
	\param n interval range
	\param seed initial state of the random number generator

	\a x needs is assumed contiguous, pre-allocated with size \a n. 
  */
  extern void fff_permutation(unsigned int* x, unsigned int n, unsigned long magic);


  /*
    \brief Generate a random combination of \a k elements in \a [0..n-1].   
 
    \a x must be contiguous, pre-allocated with size \a k. By
    convention, elements are output in ascending order.
  */
  extern void fff_combination(unsigned int* x, unsigned int k, unsigned int n, unsigned long magic); 

#ifdef __cplusplus
}
#endif

#endif

