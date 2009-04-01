/*!
  \file fff_blas.h
  \brief lite wrapper around the Fortran Basic Linear Algeabra Library (BLAS) 
  \author Alexis Roche
  \date 2008

  This library can be linked against the standard (Fortran) blas
  library, but not against cblas. 
*/

#ifndef FFF_BLAS
#define FFF_BLAS
 
#ifdef __cplusplus
extern "C" {
#endif

#include "fff_vector.h"
#include "fff_matrix.h"

#define CBLAS_INDEX_t size_t  /* this may vary between platforms */

  typedef enum {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER_t;
  typedef enum {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE_t;
  typedef enum {CblasUpper=121, CblasLower=122} CBLAS_UPLO_t;
  typedef enum {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG_t;
  typedef enum {CblasLeft=141, CblasRight=142} CBLAS_SIDE_t;
  
  /* BLAS 1 */
  extern double fff_blas_ddot (const fff_vector * x, const fff_vector * y); 
  extern double fff_blas_dnrm2 (const fff_vector * x); 
  extern double fff_blas_dasum (const fff_vector * x);
  extern CBLAS_INDEX_t fff_blas_idamax (const fff_vector * x);
  extern int fff_blas_dswap (fff_vector * x, fff_vector * y); 
  extern int fff_blas_dcopy (const fff_vector * x, fff_vector * y); 
  extern int fff_blas_daxpy (double alpha, const fff_vector * x, fff_vector * y); 
  extern int fff_blas_dscal (double alpha, fff_vector * x); 
  extern int fff_blas_drot (fff_vector * x, fff_vector * y, double c, double s); 
  extern int fff_blas_drotg (double a[], double b[], double c[], double s[]);
  extern int fff_blas_drotmg (double d1[], double d2[], double b1[], double b2, double P[]); 
  extern int fff_blas_drotm (fff_vector * x, fff_vector * y, const double P[]); 

  /* BLAS 2 */
  extern int fff_blas_dgemv (CBLAS_TRANSPOSE_t TransA, double alpha, 
			     const fff_matrix * A, const fff_vector * x, double beta, fff_vector * y); 
  extern int fff_blas_dtrmv (CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA, CBLAS_DIAG_t Diag, 
			     const fff_matrix * A, fff_vector * x); 
  extern int fff_blas_dtrsv (CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA, CBLAS_DIAG_t Diag,
			     const fff_matrix * A, fff_vector * x); 
  extern int fff_blas_dsymv (CBLAS_UPLO_t Uplo, 
			     double alpha, const fff_matrix * A, 
			     const fff_vector * x, double beta, fff_vector * y); 
  extern int fff_blas_dger (double alpha, const fff_vector * x, const fff_vector * y, fff_matrix * A);
  extern int fff_blas_dsyr (CBLAS_UPLO_t Uplo, double alpha, const fff_vector * x, fff_matrix * A); 
  extern int fff_blas_dsyr2 (CBLAS_UPLO_t Uplo, double alpha, 
			     const fff_vector * x, const fff_vector * y, fff_matrix * A); 


  /* BLAS 3 */ 
  extern int fff_blas_dgemm (CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB, 
			     double alpha, const fff_matrix * A, 
			     const fff_matrix * B, double beta, 
			     fff_matrix * C); 
  extern int fff_blas_dsymm (CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, 
			     double alpha, const fff_matrix * A, 
			     const fff_matrix * B, double beta,
			     fff_matrix * C);
  extern int fff_blas_dtrmm (CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, 
			     CBLAS_TRANSPOSE_t TransA, CBLAS_DIAG_t Diag, 
			     double alpha, const fff_matrix * A, fff_matrix * B);
  extern int fff_blas_dtrsm (CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, 
			     CBLAS_TRANSPOSE_t TransA, CBLAS_DIAG_t Diag, 
			     double alpha, const fff_matrix * A, fff_matrix * B); 
  extern int fff_blas_dsyrk (CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t Trans, 
			     double alpha, const fff_matrix * A, double beta, fff_matrix * C); 
  extern int fff_blas_dsyr2k (CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t Trans, 
			      double alpha, const fff_matrix * A, const fff_matrix * B, 
			      double beta, fff_matrix * C); 
  
  
#ifdef __cplusplus
}
#endif
 
#endif  


