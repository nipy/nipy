#include "fff_cubic_spline.h"
#include "fff_base.h"

#include <stdlib.h>


#define CUBIC_SPLINE_MIRROR(x, n, p) \
  ( (x)<0.0 ? (-(x)) : ( (x)>(n) ? ((p)-(x)) : (x) ) )

static void _fff_cubic_spline_transform( fff_vector* res, void* par ); 

static double _fff_cubic_spline_sample1d ( double x, double *coef, int dim, int stride );
static double _fff_cubic_spline_sample2d ( double x, double y, double *imcoef, int dimX, int dimY, int offX, int offY );
static double _fff_cubic_spline_sample3d ( double x, double y, double z, double *imcoef, 
					   int dimX, int dimY, int dimZ, int offX, int offY, int offZ );
static double _fff_cubic_spline_sample4d ( double x, double y, double z, double t, double *imcoef,
					   int dimX, int dimY, int dimZ, int dimT,
					   int offX, int offY, int offZ, int offT );


/* Returns the value of the cubic B-spline function at x */
double fff_cubic_spline_basis ( double x )
{

  double y, absx, aux;

  absx = FFF_ABS(x);

  if ( absx >= 2 ) 
    return 0.0;

  if ( absx < 1 ) {
    aux = absx*absx;
    y = 0.66666666666667 - aux + 0.5*absx*aux;
  }  
  else {
    aux = 2 - absx;
    y = aux*aux*aux / 6.0;
  }
 
  return y;
}


static void _fff_cubic_spline_transform( fff_vector* res, void* par ) 
{
  fff_vector *src = (fff_vector*)par; 
  
  fff_vector_memcpy( src, res );
  fff_cubic_spline_transform ( res, src ); 

  return; 
}

void fff_cubic_spline_transform ( fff_vector* res_vect, const fff_vector* src_vect )
{
  int k, dim = src_vect->size; 
  int res_stride = res_vect->stride, src_stride = src_vect->stride; 
  double cp, cm, z1_k;
  double *res = res_vect->data, *src = src_vect->data; 
  double *buf_src, *buf_res;
  const double z1 = -0.26794919243112; /* -2 + sqrt(3) */
  const double cz1 = 0.28867513459481; /* z1/(z1^2-1) */

  /* Check vector size */ 
  if (res_vect->size != dim) 
    return;

  /* 
     Initial value for the causal recursion.
     We use a mirror symmetric boundary condition for the discrete signal,
     yielding:
     
     cp(0) = (1/2-z1^(2N-2)) \sum_{k=0}^{2N-3} s(k) z1^k s(k),
     
     where we set: s(N)=s(N-2), s(N+1)=s(N-3), ..., s(2N-3)=s(1).
  */

  buf_src = src;
  cp = *buf_src;
  z1_k = 1;
  for ( k=1; k<dim; k++ ) {
    z1_k = z1 * z1_k;   /* == z1^k */
    buf_src += src_stride;            /* points towards s[k] */
    cp += (*buf_src) * z1_k;
  }

  /* At this point, we have: z1_k = z1^(N-1) */
  for ( k=2; k<dim; k++ ) {
    z1_k = z1 * z1_k;  
    buf_src -= src_stride;
    cp += (*buf_src) * z1_k;
  }
  
  /* At this point, we have: z1_k = z1^(2N-3) */
  z1_k = z1 * z1_k;
  cp = cp / ( 1 - z1_k );


  /* Storing the first causal coefficient */ 
  buf_res = res;
  *buf_res = cp;

  /* Do the causal recursion : [0..N-2]*/
  buf_src = src;
  for ( k=1; k<dim; k++ ) {
    buf_src += src_stride; 
    cp = *buf_src + z1 * cp;
    buf_res += res_stride;
    *buf_res = cp;
  }

  /* Initial value for the anticausal recursion */
  cm = cz1 * ( 2.0 * cp - *buf_src );
  *buf_res = 6.0 * cm;
 
  /* Do the anti causal recursion : [N-2..0] */
  /* for ( k=(dim-2); ((int)k)>=0; k-- ) { */
  for ( k=1; k<dim; k++ ) {
    buf_res -= res_stride;     /* buf_res points towards the k-th index */
    cm = z1 * ( cm - *buf_res );
    *buf_res = 6.0 * cm;
  }

  return;
}

/*
  work needs be allocated with at least the maximum dimension of "res" 
*/
void fff_cubic_spline_transform_image ( fff_array* res, const fff_array* src, fff_vector* work )
{
  int axis; 
  fff_vector v; 

  if ( res->datatype != FFF_DOUBLE )  {
    FFF_WARNING("Aborting. Output image encoding type must be double."); 
    return;
  }  
  if ( ( res->dimX != src->dimX ) ||
       ( res->dimY != src->dimY ) ||
       ( res->dimZ != src->dimZ ) ||
       ( res->dimT != src->dimT ) ) {
    FFF_WARNING("Aborting. Inconsistent dimensions between input and output.");     
    return;
  }

  /* Start with copying the src to the res image, forcing conversion to DOUBLE */ 
  fff_array_copy( res, src ); 

  /* Apply separable cubic spline transforms */ 
  for ( axis=0; axis<res->ndims; axis ++ ) {
    v = fff_vector_view( work->data, fff_array_dim(res, axis), work->stride ); 
    fff_array_iterate_vector_function( res, axis, &_fff_cubic_spline_transform, (void*)(&v) );
  }

  return; 
}


double fff_cubic_spline_sample ( double x, const fff_vector* coef )
{
  double res; 
  res = _fff_cubic_spline_sample1d ( x, coef->data, coef->size, coef->stride );
  return res; 
}



double fff_cubic_spline_sample_image ( double x, double y, double z, double t, const fff_array* coef )
{
  double res = 0.0; 
  double* buf = (double*)coef->data; 

  /* Check format */ 
  if ( coef->datatype != FFF_DOUBLE ) {
    FFF_WARNING("Coeff image encoding type must be double.");  
    return 0.0; 
  }


  switch( coef->ndims ) {

  case FFF_ARRAY_2D:
    res = _fff_cubic_spline_sample2d ( x, y, buf, 
				       coef->dimX, coef->dimY, 
				       coef->offsetX, coef->offsetY );
    break;
    
  case FFF_ARRAY_3D:
    res = _fff_cubic_spline_sample3d ( x, y, z, buf,
				       coef->dimX, coef->dimY, coef->dimZ,
				       coef->offsetX, coef->offsetY, coef->offsetZ );
    break;
    
  case FFF_ARRAY_4D:
    res = _fff_cubic_spline_sample4d ( x, y, z, t, buf,
				       coef->dimX, coef->dimY, coef->dimZ, coef->dimT, 
				       coef->offsetX, coef->offsetY, coef->offsetZ, coef->offsetT );
    break;

  default:
    break;

  }

  return res; 
}



/* 

Assumes: -(dimX-1) <= x <= 2*(dimX-1) 
and similarly for other coordinates. 

Returns 0 otherwise. 

*/


double _fff_cubic_spline_sample1d ( double x, double *coef, int dim, int stride )
{

  const int ddim = dim-1;
  const int two_ddim = 2*ddim;

  double *buf;
  int nx, px, xx;
  double s, aux;
  double bspx[4];
  int posx[4];
  double *buf_bspx;
  int *buf_posx;

  /* Right up superior point */
  aux = x + ddim; 
  if ( (aux<0) || (aux>3*ddim) ) 
    return 0.0;
  px = (int)(aux+2) - ddim;

  /* Left down inferior point */
  nx = px - 3;
  
  /* Compute the B-spline values as well as the image positions 
     where to find the B-spline coefficients (including mirror conditions) */ 
  buf_bspx = (double*)bspx;
  buf_posx = (int*)posx;
  for ( xx = nx; xx <= px; xx ++, buf_bspx ++, buf_posx ++ ) {
    *buf_bspx = fff_cubic_spline_basis( x-(double)xx );
    *buf_posx = CUBIC_SPLINE_MIRROR( xx, ddim, two_ddim );
  }

  /* Compute the interpolated value incrementally */
  s = 0.0;
  buf_bspx = (double*)bspx;
  buf_posx = (int*)posx;
  
  for ( xx = nx; xx <= px; xx ++, buf_bspx ++, buf_posx ++ ) {
    
    /* Point towards the coefficient value at position xx */
    buf = coef + (*buf_posx)*stride;
    
    /* Update signal value */
    s += (*buf) * (*buf_bspx);
    
  }
    
  return s;
  
}




static double _fff_cubic_spline_sample2d ( double x, double y, 
					   double *imcoef,
					   int dimX, int dimY, 
					   int offX, int offY )
{
  const int ddimX = dimX-1;
  const int ddimY = dimY-1;
  const int two_ddimX = 2*ddimX;
  const int two_ddimY = 2*ddimY;

  double *buf;
  int nx, ny, px, py, xx, yy;
  double s, aux;
  double bspx[4], bspy[4];
  int posx[4], posy[4];
  double *buf_bspx, *buf_bspy;
  int *buf_posx, *buf_posy;
  int shfty;


  /* Right up superior point */
  aux = x + ddimX; 
  if ( (aux<0) || (aux>3*ddimX) ) 
    return 0.0;
  px = (int)(aux+2) - ddimX;

  aux = y + ddimY; 
  if ( (aux<0) || (aux>3*ddimY) ) 
    return 0.0;
  py = (int)(aux+2) - ddimY;


  /* Left down inferior point */
  nx = px - 3;
  ny = py - 3;

  /* Compute the B-spline values as well as the image positions 
     where to find the B-spline coefficients (including mirror conditions) */ 
  buf_bspx = (double*)bspx;
  buf_posx = (int*)posx;
  for ( xx = nx; xx <= px; xx ++, buf_bspx ++, buf_posx ++ ) {
    *buf_bspx = fff_cubic_spline_basis( x-(double)xx );
    *buf_posx = CUBIC_SPLINE_MIRROR( xx, ddimX, two_ddimX );
  }

  buf_bspy = (double*)bspy;
  buf_posy = (int*)posy;
  for ( yy = ny; yy <= py; yy ++, buf_bspy ++, buf_posy ++ ) {
    *buf_bspy = fff_cubic_spline_basis( y-(double)yy );
    *buf_posy = CUBIC_SPLINE_MIRROR( yy, ddimY, two_ddimY );
  }


  /* Compute the interpolated value incrementally */
  s = 0.0;
  buf_bspy = (double*)bspy;
  buf_posy = (int*)posy;
    
  for ( yy = ny; yy <= py; yy ++, buf_bspy ++, buf_posy ++ ) {
    
    aux = 0.0;
    buf_bspx = (double*)bspx;
    buf_posx = (int*)posx;
    shfty = offY*(*buf_posy);

    for ( xx = nx; xx <= px; xx ++, buf_bspx ++, buf_posx ++ ) {
    
      /* Point towards the coefficient value at position (xx, yy, zz) */
      buf = imcoef + offX*(*buf_posx) + shfty;

      /* Update signal value */
      aux += (*buf) * (*buf_bspx);
    
    }
    
    s += aux * (*buf_bspy); 

  }
  
  return s;

}


static double _fff_cubic_spline_sample3d ( double x, double y, double z,
					   double *imcoef,
					   int dimX, int dimY, int dimZ, 
					   int offX, int offY, int offZ )
{
  
  const int ddimX = dimX-1;
  const int ddimY = dimY-1;
  const int ddimZ = dimZ-1;
  const int two_ddimX = 2*ddimX;
  const int two_ddimY = 2*ddimY;
  const int two_ddimZ = 2*ddimZ;

  double *buf;
  int nx, ny, nz, px, py, pz;
  int xx, yy, zz;
  double s, aux, aux2;
  double bspx[4], bspy[4], bspz[4]; 
  int posx[4], posy[4], posz[4];
  double *buf_bspx, *buf_bspy, *buf_bspz;
  int *buf_posx, *buf_posy, *buf_posz;
  int shftyz, shftz;

  /* Right up superior point */
  aux = x + ddimX; 
  if ( (aux<0) || (aux>3*ddimX) ) 
    return 0.0;
  px = (int)(aux+2) - ddimX;

  aux = y + ddimY; 
  if ( (aux<0) || (aux>3*ddimY) ) 
    return 0.0;
  py = (int)(aux+2) - ddimY;

  aux = z + ddimZ; 
  if ( (aux<0) || (aux>3*ddimZ) ) 
    return 0.0;
  pz = (int)(aux+2) - ddimZ;

  /* Left down inferior point */
  nx = px - 3;
  ny = py - 3;
  nz = pz - 3;
  
  /* Compute the B-spline values as well as the image positions 
     where to find the B-spline coefficients (including mirror conditions) */ 
  buf_bspx = (double*)bspx;
  buf_posx = (int*)posx;
  for ( xx = nx; xx <= px; xx ++, buf_bspx ++, buf_posx ++ ) {
    *buf_bspx = fff_cubic_spline_basis( x-(double)xx );
    *buf_posx = CUBIC_SPLINE_MIRROR( xx, ddimX, two_ddimX );
  }

  buf_bspy = (double*)bspy;
  buf_posy = (int*)posy;
  for ( yy = ny; yy <= py; yy ++, buf_bspy ++, buf_posy ++ ) {
    *buf_bspy = fff_cubic_spline_basis( y-(double)yy );
    *buf_posy = CUBIC_SPLINE_MIRROR( yy, ddimY, two_ddimY );
  }

  buf_bspz = (double*)bspz;
  buf_posz = (int*)posz;
  for ( zz = nz; zz <= pz; zz ++, buf_bspz ++, buf_posz ++ ) {
    *buf_bspz = fff_cubic_spline_basis( z-(double)zz );
    *buf_posz = CUBIC_SPLINE_MIRROR( zz, ddimZ, two_ddimZ );
  }

  /* Compute the interpolated value incrementally */
  s = 0.0;
  buf_bspz = (double*)bspz;
  buf_posz = (int*)posz;

  for ( zz = nz; zz <= pz; zz ++, buf_bspz ++, buf_posz ++ ) {

    aux2 = 0.0;
    buf_bspy = (double*)bspy;
    buf_posy = (int*)posy;
    shftz = offZ*(*buf_posz);
    
    for ( yy = ny; yy <= py; yy ++, buf_bspy ++, buf_posy ++ ) {
      
      aux = 0.0;
      buf_bspx = (double*)bspx;
      buf_posx = (int*)posx;
      shftyz = offY*(*buf_posy) + shftz;

      for ( xx = nx; xx <= px; xx ++, buf_bspx ++, buf_posx ++ ) {
	
	/* Point towards the coefficient value at position (xx, yy, zz) */
	buf = imcoef + offX*(*buf_posx) + shftyz;
	
	/* Update signal value */
	aux += (*buf) * (*buf_bspx);
	
      } /* end loop on x */
      aux2 += aux * (*buf_bspy); 

    }  /* end loop on y */
    s += aux2 * (*buf_bspz); 
    
  } /* end loop on z */
  

  return s;

}



static double _fff_cubic_spline_sample4d ( double x, double y, double z, double t,
					   double *imcoef,
					   int dimX, int dimY, int dimZ, int dimT, 
					   int offX, int offY, int offZ, int offT )
{
  
  const int ddimX = dimX-1;
  const int ddimY = dimY-1;
  const int ddimZ = dimZ-1;
  const int ddimT = dimT-1;
  const int two_ddimX = 2*ddimX;
  const int two_ddimY = 2*ddimY;
  const int two_ddimZ = 2*ddimZ;
  const int two_ddimT = 2*ddimT;

  double *buf;
  int nx, ny, nz, nt, px, py, pz, pt;
  int xx, yy, zz, tt;
  double s, aux, aux2, aux3;
  double bspx[4], bspy[4], bspz[4], bspt[4]; 
  int posx[4], posy[4], posz[4], post[4];
  double *buf_bspx, *buf_bspy, *buf_bspz, *buf_bspt;
  int *buf_posx, *buf_posy, *buf_posz, *buf_post;
  int shftyzt, shftzt, shftt;


  /* Right up superior point */
  aux = x + ddimX; 
  if ( (aux<0) || (aux>3*ddimX) ) 
    return 0.0;
  px = (int)(aux+2) - ddimX;

  aux = y + ddimY; 
  if ( (aux<0) || (aux>3*ddimY) ) 
    return 0.0;
  py = (int)(aux+2) - ddimY;

  aux = z + ddimZ; 
  if ( (aux<0) || (aux>3*ddimZ) ) 
    return 0.0;
  pz = (int)(aux+2) - ddimZ;

  aux = t + ddimT; 
  if ( (aux<0) || (aux>3*ddimT) ) 
    return 0.0;
  pt = (int)(aux+2) - ddimT;

  /* Left down inferior point */
  nx = px - 3;
  ny = py - 3;
  nz = pz - 3;
  nt = pt - 3;
  
  /* Compute the B-spline values as well as the image positions 
     where to find the B-spline coefficients (including mirror conditions) */ 
  buf_bspx = (double*)bspx;
  buf_posx = (int*)posx;
  for ( xx = nx; xx <= px; xx ++, buf_bspx ++, buf_posx ++ ) {
    *buf_bspx = fff_cubic_spline_basis( x-(double)xx );
    *buf_posx = CUBIC_SPLINE_MIRROR( xx, ddimX, two_ddimX );
  }

  buf_bspy = (double*)bspy;
  buf_posy = (int*)posy;
  for ( yy = ny; yy <= py; yy ++, buf_bspy ++, buf_posy ++ ) {
    *buf_bspy = fff_cubic_spline_basis( y-(double)yy );
    *buf_posy = CUBIC_SPLINE_MIRROR( yy, ddimY, two_ddimY );
  }

  buf_bspz = (double*)bspz;
  buf_posz = (int*)posz;
  for ( zz = nz; zz <= pz; zz ++, buf_bspz ++, buf_posz ++ ) {
    *buf_bspz = fff_cubic_spline_basis( z-(double)zz );
    *buf_posz = CUBIC_SPLINE_MIRROR( zz, ddimZ, two_ddimZ );
  }

  buf_bspt = (double*)bspt;
  buf_post = (int*)post;
  for ( tt = nt; tt <= pt; tt ++, buf_bspt ++, buf_post ++ ) {
    *buf_bspt = fff_cubic_spline_basis( t-(double)tt );
    *buf_post = CUBIC_SPLINE_MIRROR( tt, ddimT, two_ddimT );
  }
  

  /* Compute the interpolated value incrementally by visiting the neighbors in turn  */
  s = 0.0;
  buf_bspt = (double*)bspt;
  buf_post = (int*)post;
  
  for ( tt = nt; tt <= pt; tt ++, buf_bspt ++, buf_post ++ ) {
    
    aux3 = 0.0;
    buf_bspz = (double*)bspz;
    buf_posz = (int*)posz;
    shftt = offT*(*buf_post);
    
    for ( zz = nz; zz <= pz; zz ++, buf_bspz ++, buf_posz ++ ) {
      
      aux2 = 0.0;
      buf_bspy = (double*)bspy;
      buf_posy = (int*)posy;
      shftzt =  offZ*(*buf_posz) + shftt;
      
      for ( yy = ny; yy <= py; yy ++, buf_bspy ++, buf_posy ++ ) {
	
	aux = 0.0;
	buf_bspx = (double*)bspx;
	buf_posx = (int*)posx;
	shftyzt = offY*(*buf_posy) + shftzt;

	for ( xx = nx; xx <= px; xx ++, buf_bspx ++, buf_posx ++ ) {
	  
	  /* Point towards the coefficient value at position (xx, yy, zz, tt) */
	  buf = imcoef + offX*(*buf_posx) + shftyzt;
	  
	  /* Update signal value */
	  aux += (*buf) * (*buf_bspx);
	  
	} /* end loop on x */
	aux2 += aux * (*buf_bspy); 
	
      }  /* end loop on y */
      aux3 += aux2 * (*buf_bspz); 
      
    } /* end loop on z */
    s += aux3 * (*buf_bspt); 
    
  } /* end loop on t */
  
  return s;

}


