#include "fff_base.h"
#include "fff_matrix.h"

#include <stdlib.h>
#include <string.h>
#include <errno.h>

fff_matrix* fff_matrix_new(size_t size1, size_t size2)
{
  fff_matrix* thisone;

  thisone = (fff_matrix*)calloc(1, sizeof(fff_matrix)); 
  if (thisone == NULL) {
    FFF_ERROR("Allocation failed", ENOMEM); 
    return NULL; 
  }

  thisone->data = (double*)calloc(size1*size2, sizeof(double)); 
  if (thisone->data == NULL) 
    FFF_ERROR("Allocation failed", ENOMEM); 

  thisone->size1 = size1;
  thisone->size2 = size2;
  thisone->tda = size2;
  thisone->owner = 1; 
  
  return thisone; 
}
 

void fff_matrix_delete(fff_matrix* thisone)
{
  if (thisone->owner) 
    if (thisone->data != NULL) 
      free(thisone->data); 
  free(thisone); 
  
  return; 
}

/* View */ 
fff_matrix fff_matrix_view(const double* data, size_t size1, size_t size2, size_t tda)
{
  fff_matrix A; 

  A.size1 = size1; 
  A.size2 = size2; 
  A.tda = tda; 
  A.owner = 0; 
  A.data = (double*)data; 

  return A; 
}

/* Get element */ 
double fff_matrix_get (const fff_matrix * A, size_t i, size_t j)
{
  return(A->data[i*A->tda + j]); 
}

/* Set element */ 
void fff_matrix_set (fff_matrix * A, size_t i, size_t j, double a)
{
  A->data[i*A->tda + j] = a; 
  return;
}

/* Set all elements */ 
void fff_matrix_set_all (fff_matrix * A, double a)
{
  size_t i, j, rA;
  double *bA;
  for(i=0, rA=0; i<A->size1; i++, rA+=A->tda) {
    bA = A->data + rA;
    for(j=0; j<A->size2; j++, bA++) 
      *bA = a; 
  }
  return; 
}

/* Set all diagonal elements to a, others to zero */ 
void fff_matrix_set_scalar (fff_matrix * A, double a)
{
  size_t i, j, rA;
  double *bA;
  for(i=0, rA=0; i<A->size1; i++, rA+=A->tda) {
    bA = A->data + rA;
    for(j=0; j<A->size2; j++, bA++) {
      if (j == i) 
	*bA = a; 
      else
	*bA = 0.0; 
    }
  }
  return; 
} 

/* Global scaling */
void fff_matrix_scale (fff_matrix * A, double a)
{
  size_t i, j, rA;
  double *bA;
  for(i=0, rA=0; i<A->size1; i++, rA+=A->tda) {
    bA = A->data + rA;
    for(j=0; j<A->size2; j++, bA++) 
      *bA *= a; 
  }
  return; 
}

/* Add constant */ 
void fff_matrix_add_constant (fff_matrix * A, double a)
{
  size_t i, j, rA;
  double *bA;
  for(i=0, rA=0; i<A->size1; i++, rA+=A->tda) {
    bA = A->data + rA;
    for(j=0; j<A->size2; j++, bA++) 
      *bA += a; 
  }
  return; 
}

/* Row view */ 
fff_vector fff_matrix_row(const fff_matrix* A, size_t i)
{
  fff_vector x; 
  x.size = A->size2; 
  x.stride = 1; 
  x.owner = 0; 
  x.data = A->data + i*A->tda;
  return x; 
}

/* Column view */ 
fff_vector fff_matrix_col(const fff_matrix* A, size_t j)
{
  fff_vector x; 
  x.size = A->size1; 
  x.stride = A->tda; 
  x.owner = 0; 
  x.data = A->data + j;
  return x; 
}

/* Diagonal view */ 
fff_vector fff_matrix_diag(const fff_matrix* A)
{
  fff_vector x; 
  x.size = FFF_MIN(A->size1, A->size2); 
  x.stride = A->tda + 1; 
  x.owner = 0; 
  x.data = A->data;
  return x; 
}

/* Block view */ 
fff_matrix fff_matrix_block(const fff_matrix* A, 
			    size_t imin, size_t nrows, 
			    size_t jmin, size_t ncols) 
{
  fff_matrix Asub; 
  Asub.size1 = nrows; 
  Asub.size2 = ncols; 
  Asub.tda = A->tda; 
  Asub.owner = 0; 
  Asub.data = A->data + jmin + imin*A->tda; 
  return Asub; 
} 



/* Row copy */  
void fff_matrix_get_row (fff_vector * x, const fff_matrix * A, size_t i)
{
  fff_vector xc = fff_matrix_row(A, i); 
  fff_vector_memcpy(x, &xc); 
  return; 
}

/* Column copy */ 
void fff_matrix_get_col (fff_vector * x, const fff_matrix * A, size_t j) 
{
  fff_vector xc = fff_matrix_col(A, j); 
  fff_vector_memcpy(x, &xc); 
  return; 
}

/* Diag copy */ 
void fff_matrix_get_diag (fff_vector * x, const fff_matrix * A) 
{
  fff_vector xc = fff_matrix_diag(A); 
  fff_vector_memcpy(x, &xc); 
  return; 
}

/* Set row */ 
void fff_matrix_set_row (fff_matrix * A, size_t i, const fff_vector * x)
{
  fff_vector xc = fff_matrix_row(A, i); 
  fff_vector_memcpy(&xc, x); 
  return; 
}

/* Set column */ 
void fff_matrix_set_col (fff_matrix * A, size_t j, const fff_vector * x)
{
  fff_vector xc = fff_matrix_col(A, j); 
  fff_vector_memcpy(&xc, x); 
  return; 
}

/* Set diag */ 
void fff_matrix_set_diag (fff_matrix * A, const fff_vector * x)
{
  fff_vector xc = fff_matrix_diag(A); 
  fff_vector_memcpy(&xc, x); 
  return; 
}

/** Methods involving two matrices **/ 

#define CHECK_SIZE(A,B)						\
  if ((A->size1) != (B->size1) || (A->size2 != B->size2))		\
    FFF_ERROR("Matrices have different sizes", EDOM)

#define CHECK_TRANSPOSED_SIZE(A,B)					\
  if ((A->size1) != (B->size2) || (A->size2 != B->size1))		\
    FFF_ERROR("Incompatible matrix sizes", EDOM)

/* Copy B in A */ 
void fff_matrix_memcpy (fff_matrix * A, const fff_matrix * B)
{
  CHECK_SIZE(A, B); 

   /* If both matrices are contiguous in memory, use memcpy, otherwise
      perform a loop */ 
  if ((A->tda == A->size2) && (B->tda == B->size2))
    memcpy((void*)A->data, (void*)B->data, A->size1*A->size2*sizeof(double));
  else {
    size_t i, j, rA, rB;
    double *bA, *bB;
    for(i=0, rA=0, rB=0; i<A->size1; i++, rA+=A->tda, rB+=B->tda) {
      bA = A->data + rA;
      bB = B->data + rB;  
      for(j=0; j<A->size2; j++, bA++, bB++) 
	*bA = *bB; 
    }
  }
  
  return; 
}


/*
  Transpose a matrix: A = B**t. A needs be preallocated 

  This is equivalent to turning the matrix in
  Fortran convention (column-major order) if initially in C convention
  (row-major order), and the other way round.
*/
void fff_matrix_transpose(fff_matrix* A, const fff_matrix* B)
{
  size_t i, j, rA, rB;
  double *bA, *bB;
  CHECK_TRANSPOSED_SIZE(A, B); 
  for(i=0, rA=0, rB=0; i<A->size1; i++, rA+=A->tda) {
    bA = A->data + rA;
    bB = B->data + i;  
    for(j=0; j<A->size2; j++, bA++, bB+=B->tda) 
      *bA = *bB; 
  }
  
  return; 
}  



/* Add two matrices */ 
void fff_matrix_add (fff_matrix * A, const fff_matrix * B)
{
  size_t i, j, rA, rB;
  double *bA, *bB;
  CHECK_SIZE(A, B);
  for(i=0, rA=0, rB=0; i<A->size1; i++, rA+=A->tda, rB+=B->tda) {
    bA = A->data + rA;
    bB = B->data + rB;  
    for(j=0; j<A->size2; j++, bA++, bB++)
      *bA += *bB;    
  }
  return; 
}

/* Compute: A = A - B */ 
void fff_matrix_sub (fff_matrix * A, const fff_matrix * B)
{
  size_t i, j, rA, rB;
  double *bA, *bB;
  CHECK_SIZE(A, B);
  for(i=0, rA=0, rB=0; i<A->size1; i++, rA+=A->tda, rB+=B->tda) {
    bA = A->data + rA;
    bB = B->data + rB;  
    for(j=0; j<A->size2; j++, bA++, bB++)
      *bA -= *bB;    
  }
  return; 
}

/* Element-wise multiplication */ 
void fff_matrix_mul_elements (fff_matrix * A, const fff_matrix * B)
{
  size_t i, j, rA, rB;
  double *bA, *bB;
  CHECK_SIZE(A, B);
  for(i=0, rA=0, rB=0; i<A->size1; i++, rA+=A->tda, rB+=B->tda) {
    bA = A->data + rA;
    bB = B->data + rB;  
    for(j=0; j<A->size2; j++, bA++, bB++)
      *bA *= *bB;    
  }
  return; 
}


/* Element-wise division */ 
void fff_matrix_div_elements (fff_matrix * A, const fff_matrix * B)
{
  size_t i, j, rA, rB;
  double *bA, *bB;
  CHECK_SIZE(A, B);
  for(i=0, rA=0, rB=0; i<A->size1; i++, rA+=A->tda, rB+=B->tda) {
    bA = A->data + rA;
    bB = B->data + rB;  
    for(j=0; j<A->size2; j++, bA++, bB++)
      *bA /= *bB;    
  }
  return; 
}


long double fff_matrix_sum(const fff_matrix* A)
{
  long double sum = 0.0; 
  fff_vector a; 
  double *buf; 
  size_t i; 

  for(i=0, buf=A->data; i<A->size1; i++, buf+=A->tda) {
    a = fff_vector_view(buf, A->size2, 1); 
    sum += fff_vector_sum(&a);    
  }

  return sum; 
}
