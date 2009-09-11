/*!
  \file fff_matrix.h
  \brief fff matrix object 
  \author Alexis Roche
  \date 2003-2008

*/

#ifndef FFF_MATRIX
#define FFF_MATRIX
 
#ifdef __cplusplus
extern "C" {
#endif

#include "fff_vector.h"
#include <stddef.h>


  /*!
    \struct fff_matrix
    \brief The fff matrix structure
  */
  typedef struct {
    size_t size1; 
    size_t size2;
    size_t tda; 
    double* data; 
    int owner; 
  } fff_matrix;
  
  /*! 
    \brief fff matrix constructor
    \param size1 number of rows
    \param size2 number of columns
  */ 
  extern fff_matrix* fff_matrix_new( size_t size1, size_t size2 ); 
  /*! 
    \brief fff matrix destructor
    \param thisone instance to delete
  */ 
  extern void fff_matrix_delete( fff_matrix* thisone ); 

  extern double fff_matrix_get (const fff_matrix * A, size_t i, size_t j);
  extern void fff_matrix_set (fff_matrix * A, size_t i, size_t j, double a);
  extern void fff_matrix_set_all (fff_matrix * A, double a); 
  
  /*!
    \brief Set all diagonal elements to \a a, others to zero 
  */ 
  extern void fff_matrix_set_scalar (fff_matrix * A, double a); 
  
  extern void fff_matrix_scale (fff_matrix * A, double a); 
  extern void fff_matrix_add_constant (fff_matrix * A, double a); 

  /**
     NOT TESTED! 
   **/
  extern long double fff_matrix_sum(const fff_matrix* A); 

  /*** Views ***/ 
  extern fff_matrix fff_matrix_view(const double* data, size_t size1, size_t size2, size_t tda); 
  extern fff_vector fff_matrix_row(const fff_matrix* A, size_t i); 
  extern fff_vector fff_matrix_col(const fff_matrix* A, size_t j);
  extern fff_vector fff_matrix_diag(const fff_matrix* A);  
  extern fff_matrix fff_matrix_block(const fff_matrix* A, 
				     size_t imin, size_t nrows, 
				     size_t jmin, size_t ncols ); 

  extern void fff_matrix_get_row (fff_vector * x, const fff_matrix * A, size_t i);
  extern void fff_matrix_get_col (fff_vector * x, const fff_matrix * A, size_t j) ;
  extern void fff_matrix_get_diag (fff_vector * x, const fff_matrix * A); 
  extern void fff_matrix_set_row (fff_matrix * A, size_t i, const fff_vector * x);
  extern void fff_matrix_set_col (fff_matrix * A, size_t j, const fff_vector * x);
  extern void fff_matrix_set_diag (fff_matrix * A, const fff_vector * x); 

  extern void fff_matrix_memcpy (fff_matrix * A, const fff_matrix * B); 

  /*!
    \brief transpose a matrix
    \param B input matrix
    \param A transposed matrix on exit
    
    The matrix \c A needs be pre-allocated consistently with \c B, so
    that \c A->size1==B->size2 and \c A->size2==B->size1.
  */
  extern void fff_matrix_transpose( fff_matrix* A, const fff_matrix* B ); 

  extern void fff_matrix_add (fff_matrix * A, const fff_matrix * B);
  extern void fff_matrix_sub (fff_matrix * A, const fff_matrix * B); 
  extern void fff_matrix_mul_elements (fff_matrix * A, const fff_matrix * B);
  extern void fff_matrix_div_elements (fff_matrix * A, const fff_matrix * B);

#ifdef __cplusplus
}
#endif
 
#endif  
