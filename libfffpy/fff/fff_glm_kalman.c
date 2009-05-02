#include "fff_glm_kalman.h"
#include "fff_base.h"
#include "fff_blas.h"

#include <stdio.h>
#include <stdlib.h>

/* Declaration of static functions */
static void _fff_glm_RKF_iterate_Vb( fff_matrix* Vb, const fff_matrix* Vb0, const fff_matrix* Hspp, 
				     double aux1, double aux2, fff_matrix* Maux );
static double _fff_glm_hermit_norm( const fff_matrix* A, const fff_vector* x, fff_vector* vaux ); 



fff_glm_KF* fff_glm_KF_new( size_t dim )
{

  fff_glm_KF * thisone; 
  
  /* Start with allocating the object */
  thisone = (fff_glm_KF*) calloc( 1, sizeof(fff_glm_KF) );
  
  /* Checks that the pointer has been allocated */
  if ( thisone == NULL) 
    return NULL; 
  
  /* Allocate KF objects */
  thisone->b = fff_vector_new( dim );
  thisone->Cby = fff_vector_new( dim );
  thisone->Vb = fff_matrix_new( dim, dim ); 
  
  /* Initialization */
  thisone->dim = dim;
  thisone->t = 0;
  thisone->ssd = 0.0;
  thisone->s2 = 0.0;
  thisone->dof = 0.0;
  thisone->s2_cor = 0.0;

  /* Initialize covariance using a scalar matrix */ 
  fff_matrix_set_scalar( thisone->Vb, FFF_GLM_KALMAN_INIT_VAR); 
  
  return thisone; 
  
}


void fff_glm_KF_delete( fff_glm_KF* thisone )
{

  if ( thisone != NULL ) {
    if ( thisone->b != NULL ) fff_vector_delete(thisone->b);
    if ( thisone->Cby != NULL ) fff_vector_delete(thisone->Cby);
    if ( thisone->Vb != NULL ) fff_matrix_delete(thisone->Vb);
    free( thisone );
  }

  return;
}


void fff_glm_KF_reset( fff_glm_KF* thisone )
{
  thisone->t = 0;
  thisone->ssd = 0.0;
  thisone->s2 = 0.0;
  thisone->dof = 0.0;
  thisone->s2_cor = 0.0;
  fff_vector_set_all( thisone->b, 0.0 );
  fff_matrix_set_scalar( thisone->Vb, FFF_GLM_KALMAN_INIT_VAR );
  return;
}


void fff_glm_KF_iterate( fff_glm_KF* thisone, double y, const fff_vector* x )
{

  double Ey, Vy, invVy, ino; 

  /* Update time */ 
  thisone->t ++; 

  /* Measurement moments conditional to the effect */ 
  Ey = fff_blas_ddot( x, thisone->b ); 
  fff_blas_dsymv( CblasUpper, 1.0, thisone->Vb, x, 0.0, thisone->Cby );
  Vy = fff_blas_ddot( x, thisone->Cby ) + 1.0; 
  invVy = 1/Vy;

  /* Inovation */ 
  ino = y - Ey;
  
  /* Update effect estimate */
  fff_blas_daxpy( invVy*ino, thisone->Cby, thisone->b );

  /* Update effect variance matrix: Vb = Vb - invVy*Cby*Cby' */
  fff_blas_dger( -invVy, thisone->Cby, thisone->Cby, thisone->Vb ); 

  /* Update sum of squares and scale */
  thisone->ssd = thisone->ssd + FFF_SQR(ino)*invVy;
  thisone->s2 = thisone->ssd / (double)thisone->t; 

  return; 

}



fff_glm_RKF* fff_glm_RKF_new( size_t dim )
{

  fff_glm_RKF* thisone;
  
  /* Start with allocating the object */
  thisone = (fff_glm_RKF*) calloc( 1, sizeof(fff_glm_RKF) );
  
  /* Checks that the pointer has been allocated */
  if ( thisone == NULL) 
    return NULL; 
  
  /* Allocate RKF objects */
  thisone->Kfilt = fff_glm_KF_new( dim );
  thisone->db = fff_vector_new( dim );
  thisone->Hssd = fff_matrix_new( dim, dim );
  thisone->Gspp = fff_vector_new( dim );
  thisone->Hspp = fff_matrix_new( dim, dim ); 
  thisone->b = fff_vector_new( dim );
  thisone->Vb = fff_matrix_new( dim, dim );
  thisone->vaux = fff_vector_new( dim );
  thisone->Maux = fff_matrix_new( dim, dim );

  /* Initialization */
  thisone->dim = dim;
  thisone->t = 0;
  thisone->spp = 0.0;
  thisone->s2 = 0.0;
  thisone->a = 0.0;
  thisone->dof = 0.0;
  thisone->s2_cor = 0.0;
  
  return thisone; 
  
}
 
void fff_glm_RKF_delete( fff_glm_RKF* thisone )
{
  if ( thisone != NULL ) {
    if ( thisone->Kfilt != NULL ) fff_glm_KF_delete( thisone->Kfilt );
    if ( thisone->db != NULL ) fff_vector_delete(thisone->db);
    if ( thisone->Hssd != NULL ) fff_matrix_delete(thisone->Hssd);
    if ( thisone->Gspp != NULL ) fff_vector_delete(thisone->Gspp);
    if ( thisone->Hspp != NULL ) fff_matrix_delete(thisone->Hspp);
    if ( thisone->b != NULL ) fff_vector_delete(thisone->b);
    if ( thisone->Vb != NULL ) fff_matrix_delete(thisone->Vb);
    if ( thisone->vaux != NULL ) fff_vector_delete(thisone->vaux);
    if ( thisone->Maux != NULL ) fff_matrix_delete(thisone->Maux);
    free(thisone); 
  }
  
  return; 

}


void fff_glm_RKF_reset( fff_glm_RKF* thisone )
{
  thisone->t = 0;
  thisone->spp = 0;
  thisone->s2 = 0;
  thisone->a = 0; 
  thisone->dof = 0; 
  thisone->s2_cor = 0;

  fff_glm_KF_reset( thisone->Kfilt ); 
  fff_vector_set_all( thisone->Gspp, 0.0 );
  fff_matrix_set_all( thisone->Hssd, 0.0 );
  fff_matrix_set_all( thisone->Hspp, 0.0 );

  return;
}



void fff_glm_RKF_iterate( fff_glm_RKF* thisone, 
			  unsigned int nloop, 
			  double y, const fff_vector* x, 
			  double yy, const fff_vector* xx )
{

  unsigned int iter; 
  double cor, r, rr, ssd_ref, spp_ref, aux1, aux2;

  /* Update time */ 
  thisone->t ++; 
 
  /* Store the current OLS estimate */
  fff_vector_memcpy( thisone->vaux, thisone->Kfilt->b ); 

  /* Iterate the standard Kalman filter */
  fff_glm_KF_iterate( thisone->Kfilt, y, x ); 

  /* OLS estimate variation */
  fff_vector_memcpy( thisone->db, thisone->Kfilt->b ); 
  fff_vector_sub( thisone->db, thisone->vaux ); /* db = b - db */

  /* Update SSD hessian: Hssd = Hssd + x*x' */ 
  fff_blas_dger( 1.0, x, x, thisone->Hssd ); 

  /* Dont process any further if we are dealing with the first scan */ 
  if ( thisone->t==1 ) {
    thisone->s2 = thisone->Kfilt->s2;
    fff_vector_memcpy( thisone->b, thisone->Kfilt->b ); 
    fff_matrix_memcpy( thisone->Vb, thisone->Kfilt->Vb ); 
    return; 
  }
  /* Update bias correction factor otherwise */
  else
    cor = (double)thisone->t / (double)(thisone->t - 1); 

  /* Update SPP value */
  aux1 = fff_blas_ddot( x, thisone->Kfilt->b ); 
  r = y - aux1; 
  aux1 = fff_blas_ddot( xx, thisone->Kfilt->b );
  rr = yy - aux1;  
  aux1 = fff_blas_ddot( thisone->Gspp, thisone->db ); 
  thisone->spp += 2.0*aux1 
    + _fff_glm_hermit_norm( thisone->Hspp, thisone->db, thisone->vaux ) + r*rr;

  /* Update SPP gradient. Notice, we currently have: vaux == Hspp*db */
  fff_vector_add ( thisone->Gspp, thisone->vaux ); 
  fff_blas_daxpy( -.5*rr, x, thisone->Gspp ); 
  fff_blas_daxpy( -.5*r, xx, thisone->Gspp ); 

  /* Update SPP hessian: Hspp = Hspp + .5*(x*xx'+xx*x') */ 
  fff_blas_dsyr2( CblasUpper, .5, x, xx, thisone->Hspp ); 

  /* Update autocorrelation */
  thisone->a = cor*thisone->spp / FFF_ENSURE_POSITIVE( thisone->Kfilt->ssd );

  /* Update scale */
  thisone->s2 = thisone->Kfilt->s2;

  /* Refinement loop */
  fff_vector_memcpy( thisone->b, thisone->Kfilt->b ); 
  fff_matrix_memcpy( thisone->Vb, thisone->Kfilt->Vb );
  iter = 1; 
  while ( iter < nloop ) {

    aux1 = 1/(1 + FFF_SQR(thisone->a));
    aux2 = 2*cor*thisone->a;
    
    /* Update covariance */
    _fff_glm_RKF_iterate_Vb( thisone->Vb, thisone->Kfilt->Vb, thisone->Hspp, aux1, aux2, thisone->Maux );
    
    /* Update effect estimate */ 
    fff_blas_dsymv( CblasUpper, aux2, thisone->Vb, thisone->Gspp, 0.0, thisone->db );
    fff_vector_memcpy( thisone->b, thisone->Kfilt->b ); 
    fff_vector_add( thisone->b, thisone->db ); 
    
    /* Calculate SSD and SPP at current estimate */ 
    aux1 = fff_blas_ddot( thisone->Gspp, thisone->db ); 
    spp_ref = thisone->spp + 2*aux1 
      + _fff_glm_hermit_norm( thisone->Hspp, thisone->db, thisone->vaux );
    ssd_ref = thisone->Kfilt->ssd 
      + _fff_glm_hermit_norm( thisone->Hssd, thisone->db, thisone->vaux );
    
    /* Update autocorrelation */
    thisone->a = cor*spp_ref / FFF_ENSURE_POSITIVE(ssd_ref);
    
    /* Update scale */
    thisone->s2 = (1-FFF_SQR(thisone->a))*ssd_ref / (double)thisone->t;
    
    /* Counter */ 
    iter ++;
    
  }
  
  return; 
  
}



void fff_glm_KF_fit( fff_glm_KF* thisone,
		     const fff_vector* y,
		     const fff_matrix* X )
{
  size_t i, offset_xi = 0;
  double* yi = y->data;
  fff_vector xi;

  /* Init */ 
  fff_glm_KF_reset( thisone ); 
  xi.size = X->size2;
  xi.stride = 1;

  /* Tests */
  if ( X->size1 != y->size )
    return;

  /* Loop */
  for( i=0; i<y->size; i++, yi+=y->stride, offset_xi+=X->tda ) {
    /* Get the i-th row of the design matrix */
    xi.data = X->data + offset_xi;
    /* Iterate the Kalman filter */
    fff_glm_KF_iterate( thisone, *yi, &xi );
  }

  /* DOF */ 
  thisone->dof = (double)(y->size - X->size2); 
  thisone->s2_cor = ((double)y->size/thisone->dof)*thisone->s2; 

  return;
}



void fff_glm_RKF_fit( fff_glm_RKF* thisone,
		      unsigned int nloop, 
		      const fff_vector* y,
		      const fff_matrix* X )
{
  size_t i, offset_xi = 0;
  double* yi = y->data;
  fff_vector xi, xxi;
  double yyi = 0.0; 
  unsigned int nloop_actual = 1; 

  /* Init */ 
  fff_glm_RKF_reset( thisone ); 
  xi.size = X->size2;
  xi.stride = 1;
  xxi.size = X->size2;
  xxi.stride = 1;
  xxi.data = NULL; 

  /* Tests */
  if ( X->size1 != y->size )
    return;

  /* Loop */
  for( i=0; i<y->size; i++, yi+=y->stride, offset_xi+=X->tda ) {

    /* Get the i-th row of the design matrix */
    xi.data = X->data + offset_xi;

    /* Refinement loop only needed at the last time frame */ 
    if ( i == (y->size-1) )
      nloop_actual = nloop; 

    /* Iterate the refined Kalman filter */
    fff_glm_RKF_iterate( thisone, nloop_actual, *yi, &xi, yyi, &xxi );

    /* Copy current time values */ 
    yyi = *yi; 
    xxi.data = xi.data; 

  }

  /* DOF */ 
  thisone->dof = (double)(y->size - X->size2); 
  thisone->s2_cor = ((double)y->size/thisone->dof)*thisone->s2; 

  return;
}


/* Compute: Vb = aux1 * ( Id + aux1*aux2*Vb0*Hspp ) * Vb0
   This corresponds to a simplification as the exact update formula would be:
   Vb = aux1 * pinv( eye(p) - aux1*aux2*Vbd*He ) * Vbd
*/
static void _fff_glm_RKF_iterate_Vb( fff_matrix* Vb, const fff_matrix* Vb0, const fff_matrix* Hspp, 
				     double aux1, double aux2, fff_matrix* Maux )
{
  fff_blas_dsymm ( CblasLeft, CblasUpper, 1.0, Hspp, Vb0, 0.0, Maux ); /** Maux == Hspp*Vb0 **/ 
  fff_matrix_memcpy( Vb, Vb0 ); 
  fff_blas_dgemm( CblasNoTrans, CblasNoTrans, FFF_SQR(aux1)*aux2, Vb0, Maux, aux1, Vb ); 
  return; 
}


/* Static function to compute the Hermitian norm: x'*A*x for a
   positive symmetric matrix A. The matrix-vector product A*x is
   output in the auxiliary vector, vaux.
*/ 
static double _fff_glm_hermit_norm( const fff_matrix* A, const fff_vector* x, fff_vector* vaux )
{
  double norm = 0.0; 
  fff_blas_dsymv( CblasUpper, 1.0, A, x, 0.0, vaux );
  norm = fff_blas_ddot( x, vaux );   
  return FFF_MAX( norm, 0.0 ); 
}
