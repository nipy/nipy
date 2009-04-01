#include "fff_base.h"
#include "fff_lapack.h"

#include <errno.h>

#define FNAME FFF_FNAME

/*
dgetrf : LU decomp
dpotrf: Cholesky decomp
dgesdd: SVD decomp
dgeqrf: QR decomp
*/

#define CHECK_SQUARE(A)							\
  if ( (A->size1) != (A->size2) )					\
    FFF_ERROR("Not a square matrix", EDOM)

#define LAPACK_UPLO(Uplo) ( (Uplo)==(CblasUpper) ? "U" : "L" )


extern int FNAME(dgetrf)(int* m, int* n, double* a, int* lda, int* ipiv, int* info);
extern int FNAME(dpotrf)(char *uplo, int* n, double* a, int* lda, int* info); 
extern int FNAME(dgesdd)(char *jobz, int* m, int* n, double* a, int* lda, double* s, double* u, int* ldu,
			 double* vt, int* ldvt, double* work, int* lwork, int* iwork, int* info);
extern int FNAME(dgeqrf)(int* m, int* n, double* a, int* lda, double* tau, double* work, int* lwork, int* info);


/* Cholesky decomposition */ 
/*** Aux needs be square with the same size as A ***/
int fff_lapack_dpotrf( CBLAS_UPLO_t Uplo, fff_matrix* A, fff_matrix* Aux )
{
  char* uplo = LAPACK_UPLO(Uplo);
  int info; 
  int n = (int)A->size1; /* Assumed squared */ 
  int lda = (int)Aux->tda; 
  
  CHECK_SQUARE(A); 
  
  fff_matrix_transpose( Aux, A ); 
  FNAME(dpotrf)(uplo, &n, Aux->data, &lda, &info); 
  fff_matrix_transpose( A, Aux ); 
  
  return info; 
}

/* LU decomposition */ 
/*** Aux needs be m x n with m=A->size2 and n=A->size1 ***/
/*** ipiv needs be 1d contiguous in int with size min(m,n) ***/
int fff_lapack_dgetrf( fff_matrix* A, fff_array* ipiv, fff_matrix* Aux )
{
  int info; 
  int m = (int)A->size1; 
  int n = (int)A->size2; 
  int lda = (int)Aux->tda; 
  
  if ( (ipiv->ndims != 1) || 
       (ipiv->datatype != FFF_INT) ||
       (ipiv->dimX != FFF_MIN(m,n)) ||
       (ipiv->offsetX != 1) ) 
    FFF_ERROR("Invalid array: Ipiv", EDOM); 

  fff_matrix_transpose( Aux, A );
  FNAME(dgetrf)(&m, &n, Aux->data, &lda, (int*)ipiv->data, &info);
  fff_matrix_transpose( A, Aux ); 
  
  return info; 
}

/* QR decomposition */ 
/*** Aux needs be m x n with m=A->size2 and n=A->size1 ***/
/*** tau needs be contiguous with size min(m,n) ***/
/*** work needs be contiguous with size >= n ***/
int fff_lapack_dgeqrf( fff_matrix* A, fff_vector* tau, fff_vector* work, fff_matrix* Aux )
{
  int info; 
  int m = (int)A->size1;
  int n = (int)A->size2;
  int lda = (int)Aux->tda; 
  int lwork = (int)work->size; 

  if ( (tau->size != FFF_MIN(m,n)) ||
       (tau->stride != 1) )
    FFF_ERROR("Invalid vector: tau", EDOM); 
    
  /* Resets lwork to -1 if the input work vector is too small (in
     which case work only needs be of size >= 1) */ 
  if ( lwork < n ) 
    lwork = -1;
  else 
    if ( work->stride != 1 )  
      FFF_ERROR("Invalid vector: work", EDOM); 

  fff_matrix_transpose( Aux, A );
  FNAME(dgeqrf)(&m, &n, Aux->data, &lda, tau->data, work->data, &lwork, &info); 
  fff_matrix_transpose( A, Aux ); 
  
  return info; 
}


/* SVD decomposition */ 
/*** Aux needs be square with size max(m=A->size2, n=A->size1) ***/
/*** s needs be contiguous with size min(m,n) ***/ 
/*** U needs be m x m ***/
/*** Vt needs be n x n ***/
/*** work needs be contiguous, with size lwork such that
dmin = min(M,N)
dmax = max(M,N)

lwork >=  3*dmin**2 + max(dmax,4*dmin**2+4*dmin)

 ***/
/*** iwork needs be 1d contiguous in int with size 8*min(m,n) ***/
int fff_lapack_dgesdd( fff_matrix* A, fff_vector* s, fff_matrix* U, fff_matrix* Vt, 
		       fff_vector* work, fff_array* iwork, fff_matrix* Aux )
{
  int info; 
  int m = (int)A->size1; 
  int n = (int)A->size2; 
  int dmin = FFF_MIN(m,n); 
  int dmax = FFF_MAX(m,n); 
  int a1 = FFF_SQR(dmin);
  int a2 = 4*(a1+dmin); 
  int lwork_min = 3*a1 + FFF_MAX(dmax, a2); 
  int lda = (int)Aux->tda; 
  int ldu = (int)U->tda; 
  int ldvt = (int)Vt->tda;
  int lwork = work->size;  

  fff_matrix Aux_mm, Aux_nn; 
  
  CHECK_SQUARE(U);
  CHECK_SQUARE(Vt);
  CHECK_SQUARE(Aux);
  if ( U->size1 != m) 
    FFF_ERROR("Invalid size for U", EDOM); 
  if ( Vt->size1 != n) 
    FFF_ERROR("Invalid size for Vt", EDOM);
  if ( Aux->size1 != dmax) 
    FFF_ERROR("Invalid size for Aux", EDOM);  
  if ( (s->size != dmin) ||
       (s->stride != 1) )
    FFF_ERROR("Invalid vector: s", EDOM); 
  if ( (iwork->ndims != 1) || 
       (iwork->datatype != FFF_INT) ||
       (iwork->dimX != 8*dmin) ||
       (iwork->offsetX != 1 ) ) 
    FFF_ERROR("Invalid array: Iwork", EDOM);
  
  /* Resets lwork to -1 if the input work vector is too small (in
     which case work only needs be of size >= 1) */ 
  if ( lwork < lwork_min ) 
    lwork = -1;
  else 
    if ( work->stride != 1 )  
      FFF_ERROR("Invalid vector: work", EDOM); 
  
  /* 
     Perform the svd on A**t: 
     A**t = U* S* Vt* 
     => A = V* S* Ut*
     => U = V*, V = U*, s = s* 
     so we just need to swap m <-> n, and U <-> Vt in the input line
  */
  FNAME(dgesdd)("A", &n, &m, A->data, &lda, 
		s->data, Vt->data, &ldvt, U->data, &ldu, 
		work->data, &lwork, (int*)iwork->data, &info);

  /* At this point, both U and V are in Fortran order, so we need to
     transpose */
  Aux_mm = fff_matrix_block( Aux, 0, m, 0, m );
  fff_matrix_transpose(&Aux_mm, U); 
  fff_matrix_memcpy(U, &Aux_mm); 
  Aux_nn = fff_matrix_block( Aux, 0, n, 0, n ); 
  fff_matrix_transpose(&Aux_nn, Vt); 
  fff_matrix_memcpy(Vt, &Aux_nn); 

  return info; 
}

/* simply do the pre-allocations to simplify the use of SVD*/
static int _fff_lapack_SVD(fff_matrix* A, fff_vector* s, fff_matrix* U, fff_matrix* Vt)
{
  int n = A->size1;
  int m = A->size2;
  int dmin = FFF_MIN(m,n);
  int dmax = FFF_MAX(m,n);
  int lwork =  2* (3*dmin*dmin + FFF_MAX(dmax,4*dmin*dmin + 4*dmin));
  int liwork = 8* dmin;

  fff_vector *work = fff_vector_new(lwork);
  fff_array *iwork = fff_array_new1d(FFF_INT,liwork);
  fff_matrix *Aux = fff_matrix_new(dmax,dmax);
  
  int info = fff_lapack_dgesdd(A,s,U,Vt,work,iwork,Aux );

  fff_vector_delete(work);
  fff_array_delete(iwork);
  fff_matrix_delete(Aux);

  return info;
}

/* Compute the determinant of a symmetric matrix */
/* caveat : A is modified */
extern double fff_lapack_det_sym(fff_matrix* A)
{
  int i,n = A->size1;
  fff_matrix* U = fff_matrix_new(n,n);
  fff_matrix* Vt = fff_matrix_new(n,n);
  fff_vector* s = fff_vector_new(n);
  double det;

  _fff_lapack_SVD(A,s,U,Vt);
  for (i=0, det=1; i<n ; i++)
    det *= fff_vector_get(s,i);
  
  fff_matrix_delete(U);
  fff_matrix_delete(Vt);
  fff_vector_delete(s);
  
  return det;
}

/* Compute the inverse of a symmetric matrix */
/* caveat : A is modified */
extern int fff_lapack_inv_sym(fff_matrix* iA, fff_matrix *A)
{
  int i,n = A->size1;
  fff_matrix* U = fff_matrix_new(n,n);
  fff_matrix* Vt = fff_matrix_new(n,n);
  fff_vector* s = fff_vector_new(n);
  fff_matrix* iS = fff_matrix_new(n,n);
  fff_matrix* aux = fff_matrix_new(n,n);
  
  int info =  _fff_lapack_SVD(A,s,U,Vt);
  
  fff_matrix_set_all(iS,0);
  for (i=0 ; i<n ; i++)
	fff_matrix_set(iS,i,i,1.0/fff_vector_get(s,i));

  /* these two lines were mean to make it work with AR's bug */
  fff_blas_dgemm (CblasNoTrans, CblasNoTrans,1,U,iS, 0, aux);
  fff_blas_dgemm (CblasNoTrans, CblasTrans,1,aux,Vt,0, iA);
  
  fff_matrix_delete(U);
  fff_matrix_delete(Vt);
  fff_matrix_delete(iS);
  fff_matrix_delete(aux);
  fff_vector_delete(s);

  return info;
  
}
