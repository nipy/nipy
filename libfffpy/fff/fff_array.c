#include "fff_array.h"

#include <stdlib.h>
#include <errno.h>


/* Static functions */ 
static double _get_uchar(const char* data, size_t pos);
static double _get_schar(const char* data, size_t pos);
static double _get_ushort(const char* data, size_t pos);
static double _get_sshort(const char* data, size_t pos);
static double _get_uint(const char* data, size_t pos);
static double _get_int(const char* data, size_t pos);
static double _get_ulong(const char* data, size_t pos);
static double _get_long(const char* data, size_t pos);
static double _get_float(const char* data, size_t pos);
static double _get_double(const char* data, size_t pos);
static void _set_uchar(char* data, size_t pos, double value); 
static void _set_schar(char* data, size_t pos, double value); 
static void _set_ushort(char* data, size_t pos, double value); 
static void _set_sshort(char* data, size_t pos, double value); 
static void _set_uint(char* data, size_t pos, double value); 
static void _set_int(char* data, size_t pos, double value); 
static void _set_ulong(char* data, size_t pos, double value); 
static void _set_long(char* data, size_t pos, double value); 
static void _set_float(char* data, size_t pos, double value); 
static void _set_double(char* data, size_t pos, double value); 

static void _fff_array_iterator_update1d(void* it); 
static void _fff_array_iterator_update2d(void* it);
static void _fff_array_iterator_update3d(void* it);
static void _fff_array_iterator_update4d(void* it);
/*

Creates a C-contiguous array. 

*/ 
fff_array* fff_array_new(fff_datatype datatype,
			 size_t dimX, 
			 size_t dimY, 
			 size_t dimZ, 
			 size_t dimT)
{
  fff_array* thisone; 
  size_t nvoxels = dimX*dimY*dimZ*dimT; 
  size_t aux, offX, offY, offZ, offT;
  
  /* Offset computation */ 
  offT = 1; 
  aux = dimT; 
  offZ = aux; 
  aux *= dimZ; 
  offY = aux; 
  aux *= dimY; 
  offX = aux; 

  /* Instantiate the structure member */ 
  thisone = (fff_array*)malloc(sizeof(fff_array)); 
  if (thisone==NULL) {
    FFF_ERROR("Out of memory", ENOMEM); 
    return NULL; 
  }

  /* Set dimensions, offsets and accessors */ 
  *thisone =  fff_array_view(datatype, NULL, 
			     dimX, dimY, dimZ, dimT,
			     offX, offY, offZ, offT); 

  /* Gives ownership */ 
  thisone->owner = 1; 

  /* Allocate the image buffer */ 
  switch(datatype) {

  case FFF_UCHAR:
    {
      unsigned char* buf = (unsigned char*)calloc(nvoxels, sizeof(unsigned char)); 
      thisone->data = (void*)buf; 
    }
    break;
  case FFF_SCHAR:
    {
      signed char* buf = (signed char*)calloc(nvoxels, sizeof(signed char)); 
      thisone->data = (void*)buf; 
    }
    break;
  case FFF_USHORT:
    {
      unsigned short* buf = (unsigned short*)calloc(nvoxels, sizeof(unsigned short)); 
      thisone->data = (void*)buf;
    }
    break;
  case FFF_SSHORT:
    {
      signed short* buf = (signed short*)calloc(nvoxels, sizeof(signed short)); 
      thisone->data = (void*)buf;
    }
    break;
  case FFF_UINT:
    {
      unsigned int* buf = (unsigned int*)calloc(nvoxels, sizeof(unsigned int)); 
      thisone->data = (void*)buf; 
    }
    break;
  case FFF_INT:
    {
      int* buf = (int*)calloc(nvoxels, sizeof(int)); 
      thisone->data = (void*)buf;  
    }
    break;
  case FFF_ULONG:
    {
      unsigned long int* buf = (unsigned long int*)calloc(nvoxels, sizeof(unsigned long int)); 
      thisone->data = (void*)buf; 
    }
    break;
  case FFF_LONG:
    {
      long int* buf = (long int*)calloc(nvoxels, sizeof(long int)); 
      thisone->data = (void*)buf;
    }
    break;
  case FFF_FLOAT:
    {
      float* buf = (float*)calloc(nvoxels, sizeof(float)); 
      thisone->data = (void*)buf; 
    }
    break;
  case FFF_DOUBLE:
    {
      double* buf = (double*)calloc(nvoxels, sizeof(double)); 
      thisone->data = (void*)buf;  
    }
    break;
  default: 
    FFF_ERROR("Unrecognized data type", EINVAL);
    break; 

  }
  
  /* Report error if array has not been allocated */ 
  if (thisone->data==NULL)
    FFF_ERROR("Out of memory", ENOMEM); 

  return thisone; 
}


void fff_array_delete(fff_array* thisone)
{
  if ((thisone->owner) && (thisone->data != NULL)) 
    free(thisone->data); 
  free(thisone); 
  return; 
}


fff_array fff_array_view(fff_datatype datatype, void* buf,  
			 size_t dimX, size_t dimY, size_t dimZ, size_t dimT,
			 size_t offX, size_t offY, size_t offZ, size_t offT)
{
  fff_array thisone; 
  fff_array_ndims ndims = FFF_ARRAY_4D; 
  unsigned int nbytes = fff_nbytes(datatype);

  /* Decrease the number of dimensions if applicable */ 
  if (dimT == 1) {
    ndims = FFF_ARRAY_3D; 
    if (dimZ == 1) {
      ndims = FFF_ARRAY_2D;
      if (dimY == 1) 
	ndims = FFF_ARRAY_1D;
    }
  }
  thisone.ndims = ndims;

  /* Set dimensions / offsets / voxel size */ 
  thisone.dimX = dimX; 
  thisone.dimY = dimY; 
  thisone.dimZ = dimZ; 
  thisone.dimT = dimT; 
  thisone.offsetX = offX; 
  thisone.offsetY = offY; 
  thisone.offsetZ = offZ; 
  thisone.offsetT = offT; 
  thisone.byte_offsetX = nbytes*offX; 
  thisone.byte_offsetY = nbytes*offY; 
  thisone.byte_offsetZ = nbytes*offZ; 
  thisone.byte_offsetT = nbytes*offT; 

  /* Set data type and point towards buffer */ 
  thisone.datatype = datatype; 
  thisone.data = buf; 
  thisone.owner = 0; 

  /* Set accessors */ 
  switch(datatype) {

  case FFF_UCHAR:
    {
      thisone.get = &_get_uchar; 
      thisone.set = &_set_uchar; 
    }
    break;
  case FFF_SCHAR:
    {
      thisone.get = &_get_schar; 
      thisone.set = &_set_schar; 
    }
    break;
  case FFF_USHORT:
    {
      thisone.get = &_get_ushort; 
      thisone.set = &_set_ushort; 
    }
    break;
  case FFF_SSHORT:
    {
      thisone.get = &_get_sshort; 
      thisone.set = &_set_sshort; 
    }
    break;
  case FFF_UINT:
    {
      thisone.get = &_get_uint; 
      thisone.set = &_set_uint; 
    }
    break;
  case FFF_INT:
    {
      thisone.get = &_get_int; 
      thisone.set = &_set_int; 
    }
    break;
  case FFF_ULONG:
    {
      thisone.get = &_get_ulong; 
      thisone.set = &_set_ulong; 
    }
    break;
  case FFF_LONG:
    {
      thisone.get = &_get_long; 
      thisone.set = &_set_long; 
    }
    break;
  case FFF_FLOAT:
    {
      thisone.get = &_get_float; 
      thisone.set = &_set_float; 
    }
    break;
  case FFF_DOUBLE:
    {
      thisone.get = &_get_double; 
      thisone.set = &_set_double; 
    }
    break;
  default: 
    {
      thisone.get = NULL; 
      thisone.set = NULL; 
      FFF_ERROR("Unrecognized data type", EINVAL);
    }
    break; 

  }

  return thisone; 
}


/* Check coordinate range and return FFF_NAN if position is out of bounds */ 
double fff_array_get(const fff_array* thisone, 
		     size_t x, 
		     size_t y, 
		     size_t z, 
		     size_t t)
{
  size_t idx; 
  
  if ((x >= thisone->dimX) ||
      (y >= thisone->dimY) ||
      (z >= thisone->dimZ) ||
      (t >= thisone->dimT)) 
    return FFF_NAN; 
  
  idx = x*thisone->offsetX + y*thisone->offsetY + z*thisone->offsetZ + t*thisone->offsetT; 
  return thisone->get((const char*)thisone->data, idx); 
}


/* Check coordinate range and do noting position is out of bounds */ 
void fff_array_set(fff_array* thisone, 
		   size_t x, 
		   size_t y, 
		   size_t z, 
		   size_t t, 
		   double value)
{
  size_t idx; 
  
  if ((x >= thisone->dimX) ||
      (y >= thisone->dimY) ||
      (z >= thisone->dimZ) ||
      (t >= thisone->dimT)) 
    return;
  
  idx = x*thisone->offsetX + y*thisone->offsetY + z*thisone->offsetZ + t*thisone->offsetT; 
  thisone->set((char*)thisone->data, idx, value); 
  return; 
}



void fff_array_set_all(fff_array* thisone, double val)
{
  fff_array_iterator iter = fff_array_iterator_init(thisone); 

  while (iter.idx < iter.size) {
    fff_array_set_from_iterator(thisone, iter, val);
    fff_array_iterator_update(&iter); 
  }								
  
  return; 
}



fff_array fff_array_get_block(const fff_array* thisone, 
			      size_t x0, size_t x1, size_t fX,
			      size_t y0, size_t y1, size_t fY,
			      size_t z0, size_t z1, size_t fZ,
			      size_t t0, size_t t1, size_t fT)
{
  char* data = (char*)thisone->data; 
  data += x0*thisone->byte_offsetX + y0*thisone->byte_offsetY + z0*thisone->byte_offsetZ + t0*thisone->byte_offsetT;
  return fff_array_view(thisone->datatype, (void*)data, 
			(x1-x0)/fX+1, (y1-y0)/fY+1, (z1-z0)/fZ+1, (t1-t0)/fZ+1, 
			fX*thisone->offsetX, fY*thisone->offsetY, fZ*thisone->offsetZ, fT*thisone->offsetT); 
}



void fff_array_extrema (double* min, double* max, const fff_array* thisone) 
{
  double val; 
  fff_array_iterator iter = fff_array_iterator_init(thisone); 
  
  /* Initialization */ 
  *min = FFF_POSINF; /* 0.0;*/ 
  *max = FFF_NEGINF; /*0.0;*/ 

  while (iter.idx < iter.size) {
    val = fff_array_get_from_iterator(thisone, iter);
    if (val < *min)
      *min = val; 
    else if (val > *max) 
      *max = val;
    fff_array_iterator_update(&iter); 
  }

  return; 
}



#define CHECK_DIMS(a1,a2)					\
  if ((a1->dimX != a2->dimX) ||					\
      (a1->dimY != a2->dimY) ||					\
      (a1->dimZ != a2->dimZ) ||					\
      (a1->dimT != a2->dimT))					\
    {FFF_ERROR("Arrays have different sizes", EINVAL); return;}	\
  


void fff_array_copy(fff_array* aRes, const fff_array* aSrc)
{
  fff_array_iterator itSrc = fff_array_iterator_init(aSrc); 
  fff_array_iterator itRes = fff_array_iterator_init(aRes); 
  double valSrc; 

  CHECK_DIMS(aRes, aSrc); 
  
  while (itSrc.idx < itSrc.size) {
    valSrc = fff_array_get_from_iterator(aSrc, itSrc);
    fff_array_set_from_iterator(aRes, itRes, valSrc); 
    fff_array_iterator_update(&itSrc); 
    fff_array_iterator_update(&itRes); 
  }

  return;
}

/*
  Applies an affine correction to the input array so that:

  s0 --> r0 
  s1 --> r1

*/
void fff_array_compress(fff_array* aRes, const fff_array* aSrc, 
			double r0, double s0, 
			double r1, double s1)
{
  fff_array_iterator itSrc = fff_array_iterator_init(aSrc); 
  fff_array_iterator itRes = fff_array_iterator_init(aRes); 
  double a, b, valSrc; 

  CHECK_DIMS(aRes, aSrc); 
  
  a = (r1-r0) / (s1-s0); 
  b = r0 - a*s0; 

  while (itSrc.idx < itSrc.size) {
    valSrc = fff_array_get_from_iterator(aSrc, itSrc);
    fff_array_set_from_iterator(aRes, itRes, a*valSrc+b); 
    fff_array_iterator_update(&itSrc); 
    fff_array_iterator_update(&itRes); 
  }

  return;
}

void fff_array_add(fff_array* aRes, const fff_array* aSrc)
{
  
  fff_array_iterator itSrc = fff_array_iterator_init(aSrc); 
  fff_array_iterator itRes = fff_array_iterator_init(aRes); 
  double v; 

  CHECK_DIMS(aRes, aSrc); 

  while (itSrc.idx < itSrc.size) {
    v = fff_array_get_from_iterator(aRes, itRes); 
    v += fff_array_get_from_iterator(aSrc, itSrc); 
    fff_array_set_from_iterator(aRes, itRes, v); 
    fff_array_iterator_update(&itSrc); 
    fff_array_iterator_update(&itRes); 
  }

  return;
}

void fff_array_sub(fff_array* aRes, const fff_array* aSrc) 
{
  fff_array_iterator itSrc = fff_array_iterator_init(aSrc); 
  fff_array_iterator itRes = fff_array_iterator_init(aRes); 
  double v; 

  CHECK_DIMS(aRes, aSrc);   

  while (itSrc.idx < itSrc.size) {
    v = fff_array_get_from_iterator(aRes, itRes); 
    v -= fff_array_get_from_iterator(aSrc, itSrc); 
    fff_array_set_from_iterator(aRes, itRes, v);  
    fff_array_iterator_update(&itSrc); 
    fff_array_iterator_update(&itRes); 
  }

  return;
}

void fff_array_mul(fff_array* aRes, const fff_array* aSrc) 
{
  fff_array_iterator itSrc = fff_array_iterator_init(aSrc); 
  fff_array_iterator itRes = fff_array_iterator_init(aRes); 
  double v; 

  CHECK_DIMS(aRes, aSrc); 
  
  while (itSrc.idx < itSrc.size) {
    v = fff_array_get_from_iterator(aRes, itRes); 
    v *= fff_array_get_from_iterator(aSrc, itSrc); 
    fff_array_set_from_iterator(aRes, itRes, v);   
    fff_array_iterator_update(&itSrc); 
    fff_array_iterator_update(&itRes); 
  }

  return;
}

/*
  Force denominator's aboslute value greater than FFF_TINY. 
 */
void fff_array_div(fff_array* aRes, const fff_array* aSrc) 
{
  fff_array_iterator itSrc = fff_array_iterator_init(aSrc); 
  fff_array_iterator itRes = fff_array_iterator_init(aRes); 
  double v; 

  CHECK_DIMS(aRes, aSrc); 
  
  while (itSrc.idx < itSrc.size) {
    v = fff_array_get_from_iterator(aSrc, itSrc); 
    if (FFF_ABS(v)<FFF_TINY) 
      v = FFF_TINY; 
    v = fff_array_get_from_iterator(aRes, itRes)/v; 
    fff_array_set_from_iterator(aRes, itRes, v);   
    fff_array_iterator_update(&itSrc); 
    fff_array_iterator_update(&itRes); 
  }

  return;
}





fff_array_iterator fff_array_iterator_init_skip_axis(const fff_array* im, int axis)
{
  fff_array_iterator iter;
  size_t pY, pZ, pT;

  iter.idx = 0; 
  iter.size = im->dimX*im->dimY*im->dimZ*im->dimT; 
  
  /* Initialize pointer and coordinates */
  iter.data = (char*)im->data; 
  iter.x = 0; 
  iter.y = 0; 
  iter.z = 0; 
  iter.t = 0; 

  /* Boundary check parameters */ 
  iter.ddimY = im->dimY - 1; 
  iter.ddimZ = im->dimZ - 1; 
  iter.ddimT = im->dimT - 1;

  if (axis == 3) {
    iter.ddimT = 0;
    iter.size /= im->dimT; 
  } 
  else if (axis == 2) {
    iter.ddimZ = 0; 
    iter.size /= im->dimZ; 
  }
  else if (axis == 1) {
    iter.ddimY = 0; 
    iter.size /= im->dimY; 
  }
  else if (axis == 0) 
    iter.size /= im->dimX; 

  /* Increments */ 
  pY = iter.ddimY * im->byte_offsetY; 
  pZ = iter.ddimZ * im->byte_offsetZ; 
  pT = iter.ddimT * im->byte_offsetT; 
  iter.incT = im->byte_offsetT; 
  iter.incZ = im->byte_offsetZ - pT; 
  iter.incY = im->byte_offsetY - pZ - pT; 
  iter.incX = im->byte_offsetX - pY - pZ - pT; 
 
  /* Update function */
  switch(im->ndims) {

  case FFF_ARRAY_1D: 
    iter.update = &_fff_array_iterator_update1d; 
    break; 

  case FFF_ARRAY_2D:
    iter.update = &_fff_array_iterator_update2d; 
    break; 

  case FFF_ARRAY_3D: 
    iter.update = &_fff_array_iterator_update3d; 
    break; 

  case FFF_ARRAY_4D: 
  default:
    iter.update = &_fff_array_iterator_update4d; 
    break; 

  }
  
  return iter; 
}

fff_array_iterator fff_array_iterator_init(const fff_array* im)
{
  return fff_array_iterator_init_skip_axis(im, -1); 
}




static void _fff_array_iterator_update1d(void* it)
{
  fff_array_iterator* iter = (fff_array_iterator*)it; 
  
  iter->idx ++; 
  iter->data += iter->incX; 
  iter->x = iter->idx; 
  return; 
} 
				

static void _fff_array_iterator_update2d(void* it)
{
  fff_array_iterator* iter = (fff_array_iterator*)it; 

  iter->idx ++; 
  
  if (iter->y < iter->ddimY) {
    iter->y ++; 
    iter->data += iter->incY;
    return;
  } 
  
  iter->y = 0; 
  iter->x ++; 
  iter->data += iter->incX; 
  return; 
}



static void _fff_array_iterator_update3d(void* it)
{
  fff_array_iterator* iter = (fff_array_iterator*)it; 

  iter->idx ++; 

  if (iter->z < iter->ddimZ) {
    iter->z ++;
    iter->data += iter->incZ; 
    return; 
  }  
  
  if (iter->y < iter->ddimY) {
    iter->z = 0; 
    iter->y ++; 
    iter->data += iter->incY;
    return;
  }
  
  iter->z = 0; 
  iter->y = 0; 
  iter->x ++; 
  iter->data += iter->incX; 
  return; 
}



static void _fff_array_iterator_update4d(void* it)
{
  fff_array_iterator* iter = (fff_array_iterator*)it; 

  iter->idx ++; 

  if (iter->t < iter->ddimT) {
    iter->t ++; 
    iter->data += iter->incT; 
    return; 
  }
  
  if (iter->z < iter->ddimZ) {
    iter->t = 0; 
    iter->z ++;
    iter->data += iter->incZ; 
    return; 
  }  
  
  if (iter->y < iter->ddimY) {
    iter->t = 0; 
    iter->z = 0; 
    iter->y ++; 
    iter->data += iter->incY;
    return;
  }
  
  iter->t = 0; 
  iter->z = 0; 
  iter->y = 0; 
  iter->x ++; 
  iter->data += iter->incX; 
  return; 
}






/* Image must be in DOUBLE format */ 
void fff_array_iterate_vector_function(fff_array* im, int axis, void(*func)(fff_vector*, void*), void* par)
{
  fff_array_iterator iter; 
  fff_vector x; 

  if (im->datatype != FFF_DOUBLE) {
    FFF_WARNING("Image type must be double.");
    return; 
  }
  if ((axis>3) || (axis<0)) {
    FFF_WARNING("Invalid axis.");
    return; 
 }
   
  x.size = fff_array_dim(im, axis); 
  x.stride = fff_array_offset(im, axis); 
  x.owner = 0; 

  iter = fff_array_iterator_init_skip_axis(im, axis);
  while (iter.idx < iter.size) {
    x.data = (double*)iter.data;
    (*func)(&x, par);  
    fff_array_iterator_update(&iter); 
  }
  
  return; 
}




/* 
  Convert image values to [0,clamp-1]; typically clamp = 256.
  Possibly modify the dynamic range if the input value is
  overestimated.  For instance, the reconstructed MRI signal is
  generally encoded in 12 bits (values ranging from 0 to
  4095). Therefore, this operation may result in a loss of
  information.
*/

void fff_array_clamp(fff_array* aRes, const fff_array* aSrc, double th, int* clamp)
{
  double imin, imax, tth; 
  int dmax = *clamp - 1;
 
  /* Compute input image min and max */
  fff_array_extrema(&imin, &imax, aSrc); 

  /* Make sure the threshold is not below the min intensity */ 
  tth = FFF_MAX(th, imin);

  /* Test */ 
  if (tth>imax) {
    FFF_WARNING("Inconsistent threshold, ignored.");
    tth = imin; 
  }    

  /* If the image dynamic is small, no need for compression: just
     downshift image values and re-estimate the dynamic range (hence
     imax is translated to imax-tth casted to SSHORT) */
  if ((fff_is_integer(aSrc->datatype)) && ((imax-tth)<=dmax)) {
    fff_array_compress(aRes, aSrc, 0, tth, 1, tth+1); 
    *clamp = (int)(imax-tth) + 1;
  }
 
  /* Otherwise, compress after downshifting image values (values equal
     to the threshold are reset to zero) */ 
  else 
    fff_array_compress(aRes, aSrc, 0, tth, dmax, imax); 

  return;
}




/*************************************************************************

                    Manually templated array acessors 


 *************************************************************************/

static double _get_uchar(const char* data, size_t pos)
{
  unsigned char* buf = (unsigned char*)data;
  return((double)buf[pos]); 
}

static double _get_schar(const char* data, size_t pos)
{
  signed char* buf = (signed char*)data;
  return((double)buf[pos]); 
}

static double _get_ushort(const char* data, size_t pos)
{
  unsigned short* buf = (unsigned short*)data;
  return((double)buf[pos]); 
}

static double _get_sshort(const char* data, size_t pos)
{
  signed short* buf = (signed short*)data;
  return((double)buf[pos]); 
}

static double _get_uint(const char* data, size_t pos)
{
  unsigned int* buf = (unsigned int*)data;
  return((double)buf[pos]); 
}

static double _get_int(const char* data, size_t pos)
{
  int* buf = (int*)data;
  return((double)buf[pos]); 
}

static double _get_ulong(const char* data, size_t pos)
{
  unsigned long int* buf = (unsigned long int*)data;
  return((double)buf[pos]); 
}

static double _get_long(const char* data, size_t pos)
{
  long int* buf = (long int*)data;
  return((double)buf[pos]); 
}

static double _get_float(const char* data, size_t pos)
{
  float* buf = (float*)data;
  return((double)buf[pos]); 
}

static double _get_double(const char* data, size_t pos)
{
  double* buf = (double*)data;
  return(buf[pos]); 
}


static void _set_uchar(char* data, size_t pos, double value)
{
  unsigned char* buf = (unsigned char*)data;
  buf[pos] = (unsigned char)(FFF_ROUND(value)); 
  return; 
}

static void _set_schar(char* data, size_t pos, double value)
{
  signed char* buf = (signed char*)data;
  buf[pos] = (signed char)(FFF_ROUND(value)); 
  return; 
}

static void _set_ushort(char* data, size_t pos, double value)
{
  unsigned short* buf = (unsigned short*)data;
  buf[pos] = (unsigned short)(FFF_ROUND(value)); 
  return; 
}

static void _set_sshort(char* data, size_t pos, double value)
{
  signed short* buf = (signed short*)data;
  buf[pos] = (signed short)(FFF_ROUND(value)); 
  return; 
}

static void _set_uint(char* data, size_t pos, double value)
{
  unsigned int* buf = (unsigned int*)data;
  buf[pos] = (unsigned int)(FFF_ROUND(value)); 
  return; 
}

static void _set_int(char* data, size_t pos, double value)
{
  int* buf = (int*)data;
  buf[pos] = (int)(FFF_ROUND(value)); 
  return; 
}

static void _set_ulong(char* data, size_t pos, double value)
{
  unsigned long int* buf = (unsigned long int*)data;
  buf[pos] = (unsigned long int)(FFF_ROUND(value)); 
  return; 
}

static void _set_long(char* data, size_t pos, double value)
{
  long int* buf = (long int*)data;
  buf[pos] = (long int)(FFF_ROUND(value)); 
  return; 
}

static void _set_float(char* data, size_t pos, double value)
{
  float* buf = (float*)data;
  buf[pos] = (float)value; 
  return; 
}

static void _set_double(char* data, size_t pos, double value)
{
  double* buf = (double*)data;
  buf[pos] = value;
  return; 
}



