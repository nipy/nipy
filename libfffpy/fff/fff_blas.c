#include "fff_base.h"
#include "fff_blas.h"

#include <math.h>

#define FNAME FFF_FNAME

/* TODO : add tests for dimension compatibility */ 

/* We have to account for the fact that BLAS assumes column-major
   ordered matrices by transposing */ 

#define DIAG(Diag) ( (Diag)==(CblasUnit) ? "U" : "N" )

#define TRANS(Trans) ( (Trans)==(CblasNoTrans) ? "N" : "T" )

#define SWAP_TRANS(Trans) ( (Trans)==(CblasNoTrans) ? "T" : "N" )
#define SWAP_UPLO(Uplo) ( (Uplo)==(CblasUpper) ? "L" : "U" )
#define SWAP_SIDE(Side) ( (Side)==(CblasRight) ? "L" : "R" )



/* BLAS 1 */
extern double FNAME(ddot)(int* n, double* dx, int* incx, double* dy,
			  int* incy); 
extern double FNAME(dnrm2)(int* n, double* x, int* incx); 
extern double FNAME(dasum)(int* n, double* dx, int* incx); 
extern int FNAME(idamax)(int* n, double* dx, int* incx); 
extern int FNAME(dswap)(int* n, double* dx, int* incx,
			double* dy, int* incy); 
extern int FNAME(dcopy)(int* n, double* dx, int* incx,
			double* dy, int* incy); 
extern int FNAME(daxpy)(int* n, double* da, double* dx,
			int* incx, double* dy, int* incy);
extern int FNAME(dscal)(int* n, double* da, double* dx,
			int* incx); 
extern int FNAME(drotg)(double* da, double* db, double* c__,
			double* s);
extern int FNAME(drot)(int* n, double* dx, int* incx,
		       double* dy, int* incy, double* c__, double* s); 
extern int FNAME(drotmg)(double* dd1, double* dd2, double* 
			 dx1, double* dy1, double* dparam);
extern int FNAME(drotm)(int* n, double* dx, int* incx,
			double* dy, int* incy, double* dparam); 

/* BLAS 2 */
extern int FNAME(dgemv)(char *trans, int* m, int* n, double* 
			alpha, double* a, int* lda, double* x, int* incx,
			double* beta, double* y, int* incy);
extern int FNAME(dtrmv)(char *uplo, char *trans, char *diag, int* n,
			double* a, int* lda, double* x, int* incx); 
extern int FNAME(dtrsv)(char *uplo, char *trans, char *diag, int* n,
			double* a, int* lda, double* x, int* incx); 
extern int FNAME(dsymv)(char *uplo, int* n, double* alpha,
			double* a, int* lda, double* x, int* incx, double
			*beta, double* y, int* incy);
extern int FNAME(dger)(int* m, int* n, double* alpha,
		       double* x, int* incx, double* y, int* incy,
		       double* a, int* lda);
extern int FNAME(dsyr)(char *uplo, int* n, double* alpha,
		       double* x, int* incx, double* a, int* lda); 
extern int FNAME(dsyr2)(char *uplo, int* n, double* alpha,
			double* x, int* incx, double* y, int* incy,
			double* a, int* lda); 

/* BLAS 3 */ 
extern int FNAME(dgemm)(char *transa, char *transb, int* m, int* 
			n, int* k, double* alpha, double* a, int* lda,
			double* b, int* ldb, double* beta, double* c__,
			int* ldc); 
extern int FNAME(dsymm)(char *side, char *uplo, int* m, int* n,
			double* alpha, double* a, int* lda, double* b,
			int* ldb, double* beta, double* c__, int* ldc); 
extern int FNAME(dtrmm)(char *side, char *uplo, char *transa, char *diag,
			int* m, int* n, double* alpha, double* a, int* 
			lda, double* b, int* ldb); 
extern int FNAME(dtrsm)(char *side, char *uplo, char *transa, char *diag,
			int* m, int* n, double* alpha, double* a, int* 
			lda, double* b, int* ldb); 
extern int FNAME(dsyrk)(char *uplo, char *trans, int* n, int* k,
			double* alpha, double* a, int* lda, double* beta,
			double* c__, int* ldc); 
extern int FNAME(dsyr2k)(char *uplo, char *trans, int* n, int* k,
			 double* alpha, double* a, int* lda, double* b,
			 int* ldb, double* beta, double* c__, int* ldc); 


/****** BLAS 1 ******/ 

/* Compute the scalar product x^T y for the vectors x and y, returning the result in result.*/
double fff_blas_ddot (const fff_vector * x, const fff_vector * y)
{
  int n = (int) x->size;
  int incx = (int) x->stride; 
  int incy = (int) y->stride; 

  if ( n != y->size )
    return 1;  
 
  return( FNAME(ddot)(&n, x->data, &incx, y->data, &incy) ); 
}

/* Compute the Euclidean norm ||x||_2 = \sqrt {\sum x_i^2} of the vector x. */ 
double fff_blas_dnrm2 (const fff_vector * x)
{
  int n = (int) x->size; 
  int incx = (int) x->stride; 

  return( FNAME(dnrm2)(&n, x->data, &incx) ); 
}

/* Compute the absolute sum \sum |x_i| of the elements of the vector x.*/
double fff_blas_dasum (const fff_vector * x)
{
  int n = (int) x->size; 
  int incx = (int) x->stride; 

  return( FNAME(dasum)(&n, x->data, &incx) ); 
}

/* 
   Return the index of the largest element of the vector x. The
   largest element is determined by its absolute magnitude. We
   substract one to the original Fortran routine an actual C index.
*/ 

CBLAS_INDEX_t fff_blas_idamax (const fff_vector * x)
{
  int n = (int) x->size; 
  int incx = (int) x->stride; 

  return( (CBLAS_INDEX_t)(FNAME(idamax)(&n, x->data, &incx) - 1) ); 
}

/* Exchange the elements of the vectors x and y.*/
int fff_blas_dswap (fff_vector * x, fff_vector * y)
{
  int n = (int) x->size; 
  int incx = (int) x->stride; 
  int incy = (int) y->stride; 
  
  if ( n != y->size )
    return 1;  
  
  return( FNAME(dswap)(&n, x->data, &incx, y->data, &incy) ); 
}

/* Copy the elements of the vector x into the vector y */ 
int fff_blas_dcopy (const fff_vector * x, fff_vector * y)
{
  int n = (int) x->size; 
  int incx = (int) x->stride; 
  int incy = (int) y->stride; 
  
  if ( n != y->size )
    return 1;  
  
  return( FNAME(dcopy)(&n, x->data, &incx, y->data, &incy) ); 
}

/* Compute the sum y = \alpha x + y for the vectors x and y */ 
int fff_blas_daxpy (double alpha, const fff_vector * x, fff_vector * y)
{
  int n = (int) x->size; 
  int incx = (int) x->stride; 
  int incy = (int) y->stride; 
  
  if ( n != y->size )
    return 1;  
  
  return( FNAME(daxpy)(&n, &alpha, x->data, &incx, y->data, &incy) ); 
}

/* Rescale the vector x by the multiplicative factor alpha. */ 
int fff_blas_dscal (double alpha, fff_vector * x)
{
  int n = (int) x->size; 
  int incx = (int) x->stride; 
  
  return( FNAME(dscal)(&n, &alpha, x->data, &incx) ); 
}


/* Compute a Givens rotation (c,s) which zeroes the vector (a,b),

          [  c  s ] [ a ] = [ r ]
          [ -s  c ] [ b ]   [ 0 ]

	  The variables a and b are overwritten by the routine. */
int fff_blas_drotg (double a[], double b[], double c[], double s[])
{
  return( FNAME(drotg)(a, b, c, s) );
} 

/* Apply a Givens rotation (x', y') = (c x + s y, -s x + c y) to the vectors x, y.*/
int fff_blas_drot (fff_vector * x, fff_vector * y, double c, double s)
{
  int n = (int) x->size; 
  int incx = (int) x->stride; 
  int incy = (int) y->stride; 
  
  if ( n != y->size )
    return 1;  
  
  return( FNAME(drot)(&n, x->data, &incx, y->data, &incy, &c, &s) ); 
}

/* Compute a modified Givens transformation. The modified Givens
   transformation is defined in the original Level-1 blas
   specification. */
int fff_blas_drotmg (double d1[], double d2[], double b1[], double b2, double P[])
{
  return( FNAME(drotmg)(d1, d2, b1, &b2, P) ); 
}

    
/* Apply a modified Givens transformation.*/
int fff_blas_drotm (fff_vector * x, fff_vector * y, const double P[])
{
  int n = (int) x->size; 
  int incx = (int) x->stride; 
  int incy = (int) y->stride; 
  
  if ( n != y->size )
    return 1;  
  
  return( FNAME(drotm)(&n, x->data, &incx, y->data, &incy, (double*)P) ); 
}



/****** BLAS 2 ******/ 

/* Compute the matrix-vector product and sum y = \alpha op(A) x +
   \beta y, where op(A) = A, A^T, A^H for TransA = CblasNoTrans,
   CblasTrans, CblasConjTrans. */ 
int fff_blas_dgemv (CBLAS_TRANSPOSE_t TransA, double alpha, 
		    const fff_matrix * A, const fff_vector * x, double beta, fff_vector * y)
{
  char* trans = SWAP_TRANS(TransA); 
  int incx = (int) x->stride; 
  int incy = (int) y->stride;
  int m = (int) A->size2; 
  int n = (int) A->size1; 
  int lda = (int) A->tda; 

  return( FNAME(dgemv)(trans, &m, &n, 
		       &alpha, 
		       A->data, &lda, 
		       x->data, &incx, 
		       &beta, 
		       y->data, &incy) ); 
}


/* Compute the matrix-vector product x = op(A) x for the triangular
   matrix A, where op(A) = A, A^T, A^H for TransA = CblasNoTrans,
   CblasTrans, CblasConjTrans. When Uplo is CblasUpper then the upper
   triangle of A is used, and when Uplo is CblasLower then the lower
   triangle of A is used. If Diag is CblasNonUnit then the diagonal of
   the matrix is used, but if Diag is CblasUnit then the diagonal
   elements of the matrix A are taken as unity and are not referenced.*/ 

int fff_blas_dtrmv (CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA, CBLAS_DIAG_t Diag, 
		    const fff_matrix * A, fff_vector * x)
{
  char* uplo = SWAP_UPLO(Uplo); 
  char* trans = SWAP_TRANS(TransA); 
  char* diag = DIAG(Diag); 
  int incx = (int) x->stride; 
  int n = (int) A->size1; 
  int lda = (int) A->tda; 

  return( FNAME(dtrmv)(uplo, trans, diag, &n, 
		       A->data, &lda,
		       x->data, &incx) ); 

}

/* 
Compute inv(op(A)) x for x, where op(A) = A, A^T, A^H for TransA =
CblasNoTrans, CblasTrans, CblasConjTrans. When Uplo is CblasUpper then
the upper triangle of A is used, and when Uplo is CblasLower then the
lower triangle of A is used. If Diag is CblasNonUnit then the diagonal
of the matrix is used, but if Diag is CblasUnit then the diagonal
elements of the matrix A are taken as unity and are not referenced.
*/
int fff_blas_dtrsv (CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA, CBLAS_DIAG_t Diag,
		    const fff_matrix * A, fff_vector * x)
{
  char* uplo = SWAP_UPLO(Uplo); 
  char* trans = SWAP_TRANS(TransA); 
  char* diag = DIAG(Diag); 
  int incx = (int) x->stride; 
  int n = (int) A->size1; 
  int lda = (int) A->tda; 

  return( FNAME(dtrsv)(uplo, trans, diag, &n, 
		       A->data, &lda, 
		       x->data, &incx) ); 
}

/* 
Compute the matrix-vector product and sum y = \alpha A x + \beta y for
the symmetric matrix A. Since the matrix A is symmetric only its upper
half or lower half need to be stored. When Uplo is CblasUpper then the
upper triangle and diagonal of A are used, and when Uplo is CblasLower
then the lower triangle and diagonal of A are used.
*/

int fff_blas_dsymv (CBLAS_UPLO_t Uplo, 
		    double alpha, const fff_matrix * A, 
		    const fff_vector * x, double beta, fff_vector * y)
{
  char* uplo = SWAP_UPLO(Uplo); 
  int incx = (int) x->stride; 
  int incy = (int) y->stride;
  int n = (int) A->size1; 
  int lda = (int) A->tda; 

  return( FNAME(dsymv)(uplo, &n, 
		       &alpha, 
		       A->data, &lda, 
		       x->data, &incx, 
		       &beta, 
		       y->data, &incy) ); 
}

/* Compute the rank-1 update A = \alpha x y^T + A of the matrix A.*/
int fff_blas_dger (double alpha, const fff_vector * x, const fff_vector * y, fff_matrix * A)
{
  int incx = (int) x->stride; 
  int incy = (int) y->stride;
  int m = (int) A->size2; 
  int n = (int) A->size1; 
  int lda = (int) A->tda; 
 
  return( FNAME(dger)(&m, &n, 
		      &alpha, 
		      y->data, &incy, 
		      x->data, &incx, 
		      A->data, &lda) ); 
}

/* 
Compute the symmetric rank-1 update A = \alpha x x^T + A of the
symmetric matrix A. Since the matrix A is symmetric only its upper
half or lower half need to be stored. When Uplo is CblasUpper then the
upper triangle and diagonal of A are used, and when Uplo is CblasLower
then the lower triangle and diagonal of A are used.
*/
int fff_blas_dsyr (CBLAS_UPLO_t Uplo, double alpha, const fff_vector * x, fff_matrix * A)
{
  char* uplo = SWAP_UPLO(Uplo); 
  int incx = (int) x->stride; 
  int n = (int) A->size1; 
  int lda = (int) A->tda; 

  return( FNAME(dsyr)(uplo, &n, 
		      &alpha, 
		      x->data, &incx, 
		      A->data, &lda ) ); 
}

/* 
These functions compute the symmetric rank-2 update A = \alpha x y^T +
\alpha y x^T + A of the symmetric matrix A. Since the matrix A is
symmetric only its upper half or lower half need to be stored. When
Uplo is CblasUpper then the upper triangle and diagonal of A are used,
and when Uplo is CblasLower then the lower triangle and diagonal of A
are used.
*/
int fff_blas_dsyr2 (CBLAS_UPLO_t Uplo, double alpha, 
		    const fff_vector * x, const fff_vector * y, fff_matrix * A)
{
  char* uplo = SWAP_UPLO(Uplo); 
  int incx = (int) x->stride; 
  int incy = (int) y->stride; 
  int n = (int) A->size1; 
  int lda = (int) A->tda; 

  return( FNAME(dsyr2)(uplo, &n, 
		       &alpha, 
		       y->data, &incy, 
		       x->data, &incx, 
		       A->data, &lda) ); 
}



/****** BLAS 3 ******/ 

/*
Compute the matrix-matrix product and sum C = \alpha op(A) op(B) +
\beta C where op(A) = A, A^T, A^H for TransA = CblasNoTrans,
CblasTrans, CblasConjTrans and similarly for the parameter TransB.
*/
int fff_blas_dgemm (CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB, 
		    double alpha, const fff_matrix * A, const fff_matrix * B, double beta, fff_matrix * C)
{
  /* 
     We have A and B in C convention, hence At and Bt in F convention. 
     By computing Bt*At in F convention, we get A*B in C convention. 
     
     Hence, 
     m is the number of rows of Bt and Ct (number of cols of B and C)
     n is the number of cols of At and Ct (number of rows of A and C)
     k is the number of cols of Bt and rows of At (number of rows of B and cols of A)
  */
  char* transa = TRANS(TransA); 
  char* transb = TRANS(TransB); 
  int m = C->size2; 
  int n = C->size1;
  int lda = (int) A->tda; 
  int ldb = (int) B->tda; 
  int ldc = (int) C->tda;
  int k = (TransB == CblasNoTrans) ? (int)B->size1 : (int)B->size2;

  return( FNAME(dgemm)(transb, transa, &m, &n, &k, &alpha, 
		       B->data, &ldb, 
		       A->data, &lda, 
		       &beta, 
		       C->data, &ldc) ); 
}

/*
Compute the matrix-matrix product and sum C = \alpha A B + \beta C for
Side is CblasLeft and C = \alpha B A + \beta C for Side is CblasRight,
where the matrix A is symmetric. When Uplo is CblasUpper then the
upper triangle and diagonal of A are used, and when Uplo is CblasLower
then the lower triangle and diagonal of A are used.
*/
int fff_blas_dsymm (CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, 
		    double alpha, const fff_matrix * A, const fff_matrix * B, double beta, fff_matrix * C)
{
  char* side = SWAP_SIDE(Side); 
  char* uplo = SWAP_UPLO(Uplo); 
  int m = C->size2; 
  int n = C->size1;
  int lda = (int) A->tda; 
  int ldb = (int) B->tda; 
  int ldc = (int) C->tda; 

  return ( FNAME(dsymm)(side, uplo, &m, &n,
			&alpha, 
			A->data, &lda, 
			B->data, &ldb,
			&beta, 
			C->data, &ldc) ); 
}

/*
Compute the matrix-matrix product B = \alpha op(A) B for Side is
CblasLeft and B = \alpha B op(A) for Side is CblasRight. The matrix A
is triangular and op(A) = A, A^T, A^H for TransA = CblasNoTrans,
CblasTrans, CblasConjTrans. When Uplo is CblasUpper then the upper
triangle of A is used, and when Uplo is CblasLower then the lower
triangle of A is used. If Diag is CblasNonUnit then the diagonal of A
is used, but if Diag is CblasUnit then the diagonal elements of the
matrix A are taken as unity and are not referenced.
*/
int fff_blas_dtrmm (CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA, CBLAS_DIAG_t Diag, 
		    double alpha, const fff_matrix * A, fff_matrix * B)
{
  char* side = SWAP_SIDE(Side); 
  char* uplo = SWAP_UPLO(Uplo); 
  char* transa = TRANS(TransA); 
  char* diag = DIAG(Diag); 
  int m = B->size2; 
  int n = B->size1;
  int lda = (int) A->tda; 
  int ldb = (int) B->tda; 

  
  return( FNAME(dtrmm)(side, uplo, transa, diag, &m, &n,
		       &alpha, 
		       A->data, &lda, 
		       B->data, &ldb) ); 
  
}

/*
Compute the inverse-matrix matrix product B = \alpha op(inv(A))B for
Side is CblasLeft and B = \alpha B op(inv(A)) for Side is
CblasRight. The matrix A is triangular and op(A) = A, A^T, A^H for
TransA = CblasNoTrans, CblasTrans, CblasConjTrans. When Uplo is
CblasUpper then the upper triangle of A is used, and when Uplo is
CblasLower then the lower triangle of A is used. If Diag is
CblasNonUnit then the diagonal of A is used, but if Diag is CblasUnit
then the diagonal elements of the matrix A are taken as unity and are
not referenced.
*/
int fff_blas_dtrsm (CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA, CBLAS_DIAG_t Diag, 
		    double alpha, const fff_matrix * A, fff_matrix * B)
{
  char* side = SWAP_SIDE(Side); 
  char* uplo = SWAP_UPLO(Uplo); 
  char* transa = TRANS(TransA); 
  char* diag = DIAG(Diag); 
  int m = B->size2; 
  int n = B->size1;
  int lda = (int) A->tda; 
  int ldb = (int) B->tda; 

  return( FNAME(dtrsm)(side, uplo, transa, diag, &m, &n, 
		       &alpha, 
		       A->data, &lda, 
		       B->data, &ldb) ); 
  
}

/*
Compute a rank-k update of the symmetric matrix C, C = \alpha A A^T +
\beta C when Trans is CblasNoTrans and C = \alpha A^T A + \beta C when
Trans is CblasTrans. Since the matrix C is symmetric only its upper
half or lower half need to be stored. When Uplo is CblasUpper then the
upper triangle and diagonal of C are used, and when Uplo is CblasLower
then the lower triangle and diagonal of C are used.
*/
int fff_blas_dsyrk (CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t Trans, 
		    double alpha, const fff_matrix * A, double beta, fff_matrix * C)
{
  char* uplo = SWAP_UPLO(Uplo); 
  char* trans = SWAP_TRANS(Trans); 
  int n = C->size1;
  int k = (Trans == CblasNoTrans) ? (int)A->size1 : (int)A->size2;
  int lda = (int) A->tda; 
  int ldc = (int) C->tda; 
  
  return( FNAME(dsyrk)(uplo, trans, &n, &k,
		       &alpha, 
		       A->data, &lda, 
		       &beta,
		       C->data, &ldc) ); 
}

/* 
Compute a rank-2k update of the symmetric matrix C, C = \alpha A B^T +
\alpha B A^T + \beta C when Trans is CblasNoTrans and C = \alpha A^T B
+ \alpha B^T A + \beta C when Trans is CblasTrans. Since the matrix C
is symmetric only its upper half or lower half need to be stored. When
Uplo is CblasUpper then the upper triangle and diagonal of C are used,
and when Uplo is CblasLower then the lower triangle and diagonal of C
are used.
*/
int fff_blas_dsyr2k (CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t Trans, 
		     double alpha, const fff_matrix * A, const fff_matrix * B, double beta, fff_matrix * C)
{
  char* uplo = SWAP_UPLO(Uplo); 
  char* trans = SWAP_TRANS(Trans);
  int n = C->size1;
  int k = (Trans == CblasNoTrans) ? (int)B->size1 : (int)B->size2;
  int lda = (int) A->tda; 
  int ldb = (int) B->tda; 
  int ldc = (int) C->tda; 
 
  return( FNAME(dsyr2k)(uplo, trans, &n, &k, 
			&alpha, 
			B->data, &ldb, 
			A->data, &lda, 
			&beta, 
			C->data, &ldc) ); 
}
