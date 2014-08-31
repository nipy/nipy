/*!
  \file fff_routines.h
  \brief A few standard functions that are always necessary
  \author bertrand Thirion and Alexis Roche
  \date 2008

  Things could also be put somewhere else.
  The implementation has often a quick-and-dirty flavour.
  
*/

#ifndef FFF_ROUTINES
#define FFF_ROUTINES
 
#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <math.h>
#include "fff_array.h"
#include "fff_matrix.h"

  extern void sort_ascending_and_get_permutation( double* x, long* idx, long n );
  
  extern void sort_ascending(double *x, int n);

  extern long fff_array_argmax1d(const fff_array *farray);

  extern long fff_array_argmin1d(const fff_array *farray);
  
  extern double fff_array_min1d(const fff_array *farray);

  extern double fff_array_max1d(const fff_array *farray);


#ifdef __cplusplus
}
#endif
 
#endif  
