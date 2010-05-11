/*!
  \file fff_lapack.h
  \brief lite LAPACK wrapper
  \author Alexis Roche
  \date 2008

  fff uses the C convention for matrices (row-major order), while
  LAPACK uses the Fortran convention (column-major order). To deal
  with this issue, fff matrices need be somehow transposed, which is
  the reason why, in this interface, functions generally take an
  additional auxiliary matrix \a Aux as compared to the LAPACK
  interface.
*/

/* 
To convert a matrix \a A from C to Fortran, we may create another
matrix \a B with \a B->size1=A->size2 and \a A->size2=B->size1, then
do \a fff_matrix_transpose(B,A). Then, we may call LAPACK with \a
B->data as array input, \a m=B->size2=A->size1 rows, \a
n=B->size1=A->size2 columns and \a lda=B->tda leading dimension. The
same procedure works to perform convertion in the other way: the "C
sizes" are just the swapped "Fortan sizes".
*/

#ifndef FFF_LAPACK
#define FFF_LAPACK
 
#ifdef __cplusplus
extern "C" {
#endif

#include "fff_blas.h"
#include "fff_array.h"

  /*!
    \brief Cholesky decomposition 
    \param Uplo flag
    \param A N-by-N matrix 
    \param Aux N-by-N auxiliary matrix 

    The factorization has the form \f$ A = U^t U \f$, if \c
    Uplo==CblasUpper, or \f$ A = L L^t\f$, if \c Uplo==CblasLower,
    where \a U is an upper triangular matrix and \a L is lower
    triangular.

    On entry, if \c Uplo==CblasUpper, the leading N-by-N upper
    triangular part of \c A contains the upper triangular part of the
    matrix \a A, and the strictly lower triangular part of A is not
    referenced.  If \c Uplo==CblasLower, the leading N-by-N lower
    triangular part of \a A contains the lower triangular part of the
    matrix \a A, and the strictly upper triangular part of \a A is not
    referenced.

    On exit, \a A contains the factor \a U or \a L from the Cholesky
    factorization.
  */ 
  extern int fff_lapack_dpotrf( CBLAS_UPLO_t Uplo, fff_matrix* A, fff_matrix* Aux ); 


  /*!
    \brief LU decomposition 
    \param A M-by-N matrix 
    \param ipiv pivot indices with size min(M,N)
    \param Aux N-by-M auxiliary matrix 
    
    On entry, \a A is the M-by-N matrix to be factored.  On exit, it
    contains the factors \a L and \a U from the factorization \a
    A=PLU, where \a P is a permutation matrix, \a L is a lower
    triangular matrix with unit diagonal elements (not stored) and \a
    U is upper triangular.

    \a ipiv needs be one-dimensional contiguous in \c FFF_INT with
    size min(M,N)
  */
  extern int fff_lapack_dgetrf( fff_matrix* A, fff_array* ipiv, fff_matrix* Aux ); 

  /*!
    \brief QR decomposition 
    \param A M-by-N matrix 
    \param tau scalar factors of the elementary reflectors with size min(M,N)
    \param work auxiliary vector with size >= N 
    \param Aux N-by-M auxiliary matrix 

    Computes matrices \a Q and \a R such that \a A=QR where \a Q is
    orthonormal and \a R is triangular.

    On entry, \a A is an M-by-N matrix.  On exit, the elements on and
    above the diagonal of \a A contain the min(M,N)-by-N upper
    trapezoidal matrix \a R (\a R is upper triangular if \f$ M \geq
    N\f$); the elements below the diagonal, with the array \a tau,
    represent the orthogonal matrix \a Q as a product of min(M,N)
    reflectors. Each \a H(i) has the form

       \f$ H(i) = I - \tau  v  v^t \f$

    where \f$ \tau \f$ is a real scalar, and \a v is a real vector
    with v(1:i-1) = 0 and \a v(i)=1; \a v(i+1:M) is stored on exit in
    \a A(i+1:M,i), and \f$ \tau \f$ in \a tau(i).

    If \a work is of size 1, then the routine only computes the
    optimal size for \a work and stores the result in \c
    work->data[0]. For the actual computation, \a work should be
    contiguous with size at least N. 

    \a tau needs be contiguous as well.

    TODO: actually compute \a R using \c dorgqr. 
  */
  extern int fff_lapack_dgeqrf( fff_matrix* A, fff_vector* tau, fff_vector* work, fff_matrix* Aux );
  
  /*! 
    \brief Singular Value Decomposition
    \param A M-by-N matrix to decompose (to be overwritten)
    \param s singular values in descending order, with size min(M,N)
    \param U M-by-M matrix 
    \param Vt N-by-N matrix 
    \param work auxiliary vector
    \param iwork auxiliary array of integers 
    \param Aux auxiliary square matrix with size max(M,N)
    
    Computes a diagonal matrix \a S and orthonormal matrices \a U and
    \a Vt such that \f$ A = U S V^t \f$. 
    
    If \a work is of size 1, then the routine only computes the
    optimal size for \a work and stores the result in \c
    work->data[0]. For the actual computation, \a work should be
    contiguous with size at least: \f$ L_{work} \geq 3 d_{\min}^2 +
    \max(d_{\max}, 4 (d_{\min}^2 + d_{\min})) \f$ where \f$
    d_{\min}=\min(M,N) \f$ and \f$ d_{\max}=\max(M,N) \f$. For good
    performance, \f$ L_{work} \f$ should generally be larger.


    \a iwork needs be one-dimensional contiguous in \c FFF_INT with size 8*min(M,N)
  */

  extern int fff_lapack_dgesdd( fff_matrix* A, fff_vector* s, fff_matrix* U, fff_matrix* Vt, 
				fff_vector* work, fff_array* iwork, fff_matrix* Aux ); 

  /*
	\brief Computation of the determinant of symmetric matrices
	\param A M-by-M matrix (to be overwritten)
	
	The determinant is returned as output of the function.
	The procedure uses the SVD hence it is valid only for symmetric matrices.
	It is not meant to be optimal at the moment.
	Caveat : no check is performed -- untested version
  */

  extern double fff_lapack_det_sym(fff_matrix* A);

 /*
	\brief Computation of the inverse of  of symmetric matrices
	\param iA The resulting output matrix
	\param A M-by-M matrix to be inverted (to be overwritten)
	
	The determinant is returned as output of the function.
	The procedure uses the SVD hence it is valid only for symmetric matrices.
	It is not meant to be optimal at the moment.
	Caveat : no check is performed -- untested version
  */

	extern int fff_lapack_inv_sym(fff_matrix* iA, fff_matrix *A);

#ifdef __cplusplus
}
#endif
 
#endif  
