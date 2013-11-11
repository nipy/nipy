#include "fff_routines.h"
#include "fff_base.h"

#include <stdlib.h>
#include <stdio.h>



typedef struct{
  double x; 
  long i; 
} dummy_struct;

static int _dummy_struct_geq(const void * x, const void * y)
{
  int ans = -1; 
  dummy_struct xx = *((dummy_struct*)x);
  dummy_struct yy = *((dummy_struct*)y);

  if ( xx.x > yy.x ) { 
    ans = 1; 
    return ans; 
  }
  if ( xx.x == yy.x ) 
    ans = 0; 

  return ans;  
}

extern void sort_ascending_and_get_permutation( double* x, long* idx, long n )
{
  long i; 
  double *bufx;
  dummy_struct* xx = (dummy_struct*)calloc( n, sizeof(dummy_struct) );  
  dummy_struct* buf_xx; 
  long* buf_idx; 

  bufx = x; 
  buf_idx = idx; 
  buf_xx = xx; 
  for ( i=0; i<n; i++, bufx++, buf_xx++ ) {
    (*buf_xx).x = *bufx;
    (*buf_xx).i = i; 
  }

  qsort ( xx, n, sizeof(dummy_struct), &_dummy_struct_geq );

  bufx = x; 
  buf_idx = idx; 
  buf_xx = xx; 
  for ( i=0; i<n; i++, bufx++, buf_idx++, buf_xx++ ) {
    *bufx = (*buf_xx).x;
    *buf_idx = (*buf_xx).i;
  }

  free( xx ); 
  return; 
}

extern void sort_ascending(double *x, int n)
{
  long *idx = (long *) calloc (n,sizeof(long));
  sort_ascending_and_get_permutation( x, idx,n);
  free(idx);
}



extern long fff_array_argmax1d(const fff_array *farray)
{
  /* returns the index of the max value on a supposedly 1D array */
  /* quick and dirty implementation */
  long i,n = farray->dimX;
  long idx = 0;
  double val,max = (double) fff_array_get1d(farray,idx);
  
  for (i=0 ; i<n ; i++){
	val = (double) fff_array_get1d(farray,i);
	if (val>max){
	  max = val;
	  idx = i;
	}
  }
  return idx;
}

extern long fff_array_argmin1d(const fff_array *farray)
{
  /*
	returns the index of the max value on a supposedly 1D array 
	quick and dirty implementation 
  */
  long i,n = farray->dimX;
  long idx = 0;
  double val,min = (double) fff_array_get1d(farray,idx);
  
  for (i=0 ; i<n ; i++){
	val = (double) fff_array_get1d(farray,i);
	if (val<min){
	  min = val;
	  idx = i;
	}
  }
  return idx;
}


extern double fff_array_min1d(const fff_array *farray)
{
  /*
	returns the index of the max value on a supposedly 1D array
	quick and dirty implementation 
  */
  long i,n = farray->dimX;
  double val,min = (double) fff_array_get1d(farray,0);
  
  for (i=0 ; i<n ; i++){
	val = (double) fff_array_get1d(farray,i);
	if (val<min)
	  min = val;
  }
  return min;
}

extern double fff_array_max1d(const fff_array *farray)
{
  /*
	returns the index of the max value on a supposedly 1D array
   quick and dirty implementation
  */
  long i,n = farray->dimX;
  double val,max = (double) fff_array_get1d(farray,0);
  
  for (i=0 ; i<n ; i++){
	val = (double) fff_array_get1d(farray,i);
	if (val>max)
	  max = val;
  }
  return max;
}

