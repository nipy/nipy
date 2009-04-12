/*!
  \file fff_vector.h
  \brief fff vector object 
  \author Alexis Roche
  \date 2003-2008

*/

#ifndef FFF_VECTOR
#define FFF_VECTOR
 
#ifdef __cplusplus
extern "C" {
#endif

#include "fff_base.h"
#include <stddef.h>

 
  /*!
    \struct fff_vector
    \brief The fff vector structure
    
  */
  typedef struct {
    size_t size; 
    size_t stride; 
    double* data; 
    int owner; 
  } fff_vector;
  

  /*! 
    \brief fff vector constructor
    \param size vector size
  */ 
  extern fff_vector* fff_vector_new(size_t size); 

  /*! 
    \brief fff vector destructor
    \param thisone instance to delete
  */ 
  extern void fff_vector_delete(fff_vector* thisone); 
  /*!
    \brief Vector view
    \param data data array 
    \param size array size
    \param stride array stride
  */
  extern fff_vector fff_vector_view(const double* data, size_t size, size_t stride); 
 
  /*! 
    \brief Get an element 
    \param x vector 
    \param i index
  */ 
  extern double fff_vector_get (const fff_vector * x, size_t i); 

  /*! 
    \brief Set an element 
    \param x vector 
    \param i index
    \param a value to set 
  */ 
  extern void fff_vector_set (fff_vector * x, size_t i, double a); 

  /*! 
    \brief Set all elements to a constant value 
    \param x vector 
    \param a value to set 
  */ 
  extern void fff_vector_set_all (fff_vector * x, double a); 
  extern void fff_vector_scale (fff_vector * x, double a); 
  extern void fff_vector_add_constant (fff_vector * x, double a); 
  
  /*! 
    \brief Copy a vector 
    \param x input vector 
    \param y output vector 
  */ 
  extern void fff_vector_memcpy( fff_vector* x, const fff_vector* y );

  /*! 
    \brief view or copy an existing buffer
    \param x destination vector  
    \param data pre-allocated buffer
    \param datatype data type 
    \param stride stride in relative units (1 means contiguous array)
  */ 
  extern void fff_vector_fetch(fff_vector* x, const void* data, fff_datatype datatype, size_t stride); 


  /*! 
    \brief Add two vectors 
    \param x output vector 
    \param y constant vector 
  */ 
  extern void fff_vector_add (fff_vector * x, const fff_vector * y);

  /*!
    \brief Compute the difference x-y
    \param x output vector
    \param y constant vector
  */
  extern void fff_vector_sub (fff_vector * x, const fff_vector * y); 
  extern void fff_vector_mul (fff_vector * x, const fff_vector * y); 
  extern void fff_vector_div (fff_vector * x, const fff_vector * y); 

  /*! 
    \brief Sum up vector elements 
    \param x input vector 
  */ 
  extern long double fff_vector_sum( const fff_vector* x ); 
  /*! 
    \brief Sum of squared differences 
    \param x input vector 
    \param m offset value, either fixed or set to the mean
    \param fixed_offset true if the offset is to be held fixed

    Compute the sum: \f$ \sum_i (x_i-a)^2 \f$ where \a a is a given
    offset.
  */ 
  extern long double fff_vector_ssd( const fff_vector* x, double* m, int fixed_offset ); 
  
  extern long double fff_vector_wsum( const fff_vector* x, const fff_vector* w, long double* sumw ); 
  extern long double fff_vector_sad( const fff_vector* x, double m ); 

  /*!
    \brief Fast median from non-const vector
    \param x input vector 
  
    Beware that the input array is re-arranged.  This function does
    not require the input array to be sorted in ascending order. It
    deals itself with sorting the data, and this is done in a partial
    way, yielding a faster algorithm.
  */  
  extern double fff_vector_median( fff_vector* x );
  
  /*!
    \brief Sample percentile, or quantile from non-const array
    \param input vector 
    \param r value between 0 and 1
    \param interp interpolation flag

    If \c interp is \c FALSE, this function returns the smallest
    sample value \a q that is greater than or equal to a proportion \a
    r of all sample values; more precisely, the number of sample
    values that are greater or equal to \a q is smaller or equal to \a
    (1-r) times the sample size. If \c interp is \c TRUE, then the
    quantile is defined from a linear interpolation of the empirical
    cumulative distribution. For instance, if \a r = 0.5 and \c interp
    = \c TRUE, \a q is the usual median; the \c interp flag does not
    play any role if the sample size is odd. Similarly to \c
    fff_median_from_temp_data, the array elements are re-arranged.
  */  
  extern double fff_vector_quantile( fff_vector* x, double r, int interp );
  /*!
    \brief Weighted median
    \param x already sorted data 
    \param w weight vector
    
    Compute the weighted median of \c x_sorted using the weights in \c
    w, assuming the elements in \c x_sorted are in ascending
    order. Notice, the function does not check for negative weights;
    if the weights sum up to a negative value, \c FFF_NAN is returned.
  */  
  extern double fff_vector_wmedian_from_sorted_data ( const fff_vector* x_sorted,
						      const fff_vector* w ); 

#ifdef __cplusplus
}
#endif
 
#endif  
