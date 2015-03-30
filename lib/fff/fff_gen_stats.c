#include "fff_gen_stats.h"
#include "fff_lapack.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>

#include <stdio.h>

/*
  Generate a random permutation from [0..n-1]. 
*/
extern void fff_permutation(unsigned int* x, unsigned int n, unsigned long magic)
{ 
  unsigned int* xi, i, ir, j, tmp, nc; 
  unsigned long int m = magic; 

  /* Initialize x as the identity permutation */ 
  for(i=0, xi=x; i<n; i++, xi++) 
    *xi = i;

  /* Draw numbers iteratively and rearrange the array */ 
  for(i=0, nc=n; i<n; i++, nc--) {

    /* Draw j in range [i..n[ */
    ir = m % nc;
    m = m / nc; 
    j = ir + i; 

    /* Move x[j] to i-th index and shift indices in range [i..j[ to
       the right */ 
    tmp = x[j]; 
    xi = x + i; 
    memmove((void*)(xi+1), (void*)xi, ir*sizeof(unsigned int));
    *xi = tmp; 
    
  }
 
  return; 
}


/*
  Generate a random combination of k elements in [0..n-1]. 
  x must be pre-allocated with size k. 
*/

static unsigned long int _combinations(unsigned int k, unsigned int n)
{
  unsigned long int c, i, aux; 

  /* Compute the total number of combinations: Cn,k */ 
  aux = n - k; 
  for (i=1, c=1; i<=k; i++) {
    c *= (aux+i);
    c /= i;  
  }

  return FFF_MAX(c, 1);  
}

extern void fff_combination(unsigned int* x, unsigned int k, unsigned int n, unsigned long magic)
{ 
  unsigned long int kk, nn, i; 
  unsigned long int m = magic; 
  unsigned int *bx = x; 
  unsigned long int c; 

  /* Ensure 0 <= magic < Cn,k */ 
  c = _combinations(k, n); 
  m = magic % c; 

  /* Loop. At the beginning of each iteration, c == Cn-(i+1),k-(i+1). */
  i = 0; 
  kk = k; 
  nn = n; 
  kk = k; 
  while( kk > 0 ) {

    nn --;
    c = _combinations(kk-1, nn); 
 
    /* If i is accepted, then store it and do: kk-- */
    if ( m < c ) {
      *bx = i; 
      bx ++; 
      kk --; 
    }
    else
      m = m - c; 

     /* Next candidate */ 
    i ++; 
    
  }

  return; 
}


/* 
   Squared mahalanobis distance: d2 = x' S^-1 x 
   Beware: x is not const
*/ 
extern double fff_mahalanobis(fff_vector* x, fff_matrix* S, fff_vector* xaux, fff_matrix* Saux)
{
  double d2; 

  /* Copy x into xaux */
  fff_vector_memcpy(xaux, x);

  /* Compute: xaux = S^-1 x using Cholesky decomposition */
  fff_lapack_solve_chol(S, xaux, Saux); 
 
  /* Compute x' S^-1 x */ 
  d2 = fff_blas_ddot (x, xaux);

  return d2; 
}

