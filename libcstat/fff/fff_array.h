/*!
  \file fff_array.h
  \brief Basic image object
  \author Alexis Roche
  \date 2005-2006

  This library implements a generic 4-dimensional array object that
  can be used to represent images. 
*/

 
#ifndef FFF_ARRAY
#define FFF_ARRAY

#ifdef __cplusplus
extern "C" {
#endif

#include "fff_base.h"
#include "fff_vector.h"

#include <stddef.h>


#define fff_array_dim(array, axis)					\
  ((axis)==0 ? (array->dimX) : ((axis)==1 ? (array->dimY) : ((axis)==2 ? (array->dimZ) : (array->dimT)) ) )  
#define fff_array_offset(array, axis)					\
  ((axis)==0 ? (array->offsetX) : ((axis)==1 ? (array->offsetY) : ((axis)==2 ? (array->offsetZ) : (array->offsetT)) ) )  
  
  /*
    #define fff_array_copy(ares, asrc)		\
    fff_array_compress(ares, asrc, 0, 0, 1, 1)
  */
 
#define fff_array_new1d(dtype, dx)		\
  fff_array_new(dtype, dx, 1, 1, 1)
#define fff_array_new2d(dtype, dx, dy)		\
  fff_array_new(dtype, dx, dy, 1, 1)
#define fff_array_new3d(dtype, dx, dy, dz)	\
  fff_array_new(dtype, dx, dy, dz, 1)
  
#define fff_array_view1d(dtype, data, dx, ox)		\
  fff_array_view(dtype, data, dx, 1, 1, 1, ox, 1, 1, 1)
#define fff_array_view2d(dtype, data, dx, dy, ox, oy)		\
  fff_array_view(dtype, data, dx, dy, 1, 1, ox, oy, 1, 1)
#define fff_array_view3d(dtype, data, dx, dy, dz, ox, oy, oz)	\
  fff_array_view(dtype, data, dx, dy, dz, 1, ox, oy, oz, 1)
  
#define fff_array_get1d(array, x)		\
  fff_array_get(array, x, 0, 0, 0)
#define fff_array_get2d(array, x, y)		\
  fff_array_get(array, x, y, 0, 0)
#define fff_array_get3d(array, x, y)		\
  fff_array_get(array, x, y, z, 0)
  
#define fff_array_set1d(array, x, a)		\
  fff_array_set(array, x, 0, 0, 0, a)
#define fff_array_set2d(array, x, y, a)		\
  fff_array_set(array, x, y, 0, 0, a)
#define fff_array_set3d(array, x, y, z, a)	\
  fff_array_set(array, x, y, z, 0, a)
  
#define fff_array_get_block1d(array, x0, x1, fx)			\
  fff_array_get_block(array, x0, x1, fx, 0, 0, 1, 0, 0, 1, 0, 0, 1)
#define fff_array_get_block2d(array, x0, x1, fx, y0, y1, fy)		\
  fff_array_get_block(array, x0, x1, fx, y0, y1, fy, 0, 0, 1, 0, 0, 1)
#define fff_array_get_block3d(array, x0, x1, fx, y0, y1, fy, z0, z1, fz) \
  fff_array_get_block(array, x0, x1, fx, y0, y1, fy, z0, z1, fz, 0, 0, 1)
  

#define fff_array_get_from_iterator(array, iter)	\
  array->get(iter.data, 0) 
  
#define fff_array_set_from_iterator(array, iter, val)	\
  array->set(iter.data, 0, val) 

#define fff_array_iterator_update(iter)	\
      (iter)->update(iter)

  /*!
    \typedef fff_array_ndims
    \brief Image flag type
  */
  typedef enum {
    FFF_ARRAY_1D = 1,   /*!< 1d image */
    FFF_ARRAY_2D = 2,   /*!< 2d image */
    FFF_ARRAY_3D = 3,   /*!< 3d image */
    FFF_ARRAY_4D = 4    /*!< 4d image */
  } fff_array_ndims;

  
  /*!
    \struct fff_array
    \brief The fff image structure
    
    Image values are stored in a \c void linear array, the actual
    encoding type being specified by the field \c datatype. The image
    dimension along each axis are encoded by fields starting with \c
    dim, while the \c ndims flag specifies the biggest axis index
    corresponding to a non-unitary dimension; it essentially defines
    whether the image is 1d, 2d, 3d, or 4d. The use of offsets (or
    strides) makes the object independent from any storage
    convention. A pixel with coordinates (\a x, \a y, \a z, \a t) may
    be accessed using a command like:

    \code 
    value = im->data[ x*im->offsetX + y*im->offsetY + z*im->offsetZ + t*im->offsetT ];
    \endcode

    Note that this approach makes it possible to extract a sub-image
    from an original image without the need to reallocate memory. 
  */
  typedef struct {
    fff_array_ndims ndims; /*!< Image flag */
    fff_datatype datatype; /*!< Image encoding type */
    size_t dimX; /*!< Dimension (number of pixels) along first axis */
    size_t dimY; /*!< Dimension (number of pixels) along second axis */
    size_t dimZ; /*!< Dimension (number of pixels) along third axis */
    size_t dimT; /*!< Dimension (number of pixels) along fourth axis */
    size_t offsetX; /*!< Offset (relative to type) along first axis */
    size_t offsetY; /*!< Offset (relative to type) along second axis */
    size_t offsetZ; /*!< Offset (relative to type) along third axis */
    size_t offsetT; /*!< Offset (relative to type) along fourth axis */
    size_t byte_offsetX; /*!< Offset (in bytes) along first axis */
    size_t byte_offsetY; /*!< Offset (in bytes) along second axis */
    size_t byte_offsetZ; /*!< Offset (in bytes) along third axis */
    size_t byte_offsetT; /*!< Offset (in bytes) along fourth axis */
    void* data; /*!< Image buffer */
    int owner; /*!< Non-zero if the object owns its data */
    double (*get)(const char*, size_t); /*!< Get accessor */
    void (*set)(char*, size_t, double); /*!< Set accessor */ 
  } fff_array;


  /*!
    \struct fff_array_iterator
    \brief Image iterator structure
  */
  typedef struct {
    size_t idx;
    size_t size; 
    char* data; 
    size_t x; 
    size_t y; 
    size_t z; 
    size_t t;
    size_t ddimY; 
    size_t ddimZ; 
    size_t ddimT;
    size_t incX; 
    size_t incY; 
    size_t incZ; 
    size_t incT; 
    void (*update)(void*); /*!< Updater */
  } fff_array_iterator;
    

  /*! 
    \brief Constructor for the fff_array structure 
    \param datatype image encoding type
    \param dimX number of pixels along the first axis
    \param dimY number of pixels along the second axis
    \param dimZ number of pixels along the third axis
    \param dimT number of pixels along the fourth axis

    This function allocates a new image buffer. 
  */
  extern fff_array* fff_array_new(fff_datatype datatype,
				   size_t dimX, 
				   size_t dimY, 
				   size_t dimZ, 
				   size_t dimT); 
  
  /*!
    \brief Destructor for the \c fff_array structure
    \param thisone fff_array member to be deleted
  */
  extern void fff_array_delete(fff_array* thisone);
  
  
  /*! 
    \brief Array view
    \param datatype image encoding type
    \param buf already allocated image buffer 
    \param dimX number of pixels along the first axis
    \param dimY number of pixels along the second axis
    \param dimZ number of pixels along the third axis
    \param dimT number of pixels along the fourth axis
    \param offX offset along the first axis
    \param offY offset along the second axis
    \param offZ offset along the third axis
    \param offT offset along the fourth axis

    This function assumes that the image buffer is already allocated. 
  */
  extern fff_array fff_array_view(fff_datatype datatype, void* buf, 
				   size_t dimX, size_t dimY, size_t dimZ, size_t dimT,
				   size_t offX, size_t offY, size_t offZ, size_t offT); 


  /*! 
    \brief Generic function to access a voxel's value
    \param thisone input image
    \param x first coordinate
    \param y second coordinate
    \param z third coordinate
    \param t fourth coordinate
	
    Get image value at a specific location defined by voxel coordinates. 
    Return \c fff_NAN if the position is out of bounds. 
 */ 
  extern double fff_array_get(const fff_array* thisone, 
			      size_t x, 
			      size_t y, 
			      size_t z, 
			      size_t t); 

  /*! 
    \brief Generic function to set one voxel's value
    \param value value to set 
    \param thisone input image
    \param x first coordinate
    \param y second coordinate
    \param z third coordinate
    \param t fourth coordinate
  */ 
  extern void fff_array_set(fff_array* thisone, 
			    size_t x, 
			    size_t y, 
			    size_t z, 
			    size_t t, 
			    double value); 

  /*!
    \brief Set all pixel values to a given constant
    \param thisone image
    \param c constant
  */
  extern void fff_array_set_all(fff_array* thisone, double c); 


  /*! 
    \brief Extract an image block 
    \param thisone input image
    \param x0 first coordinate of the starting point
    \param x1 first coordinate of the finishing point
    \param y0 second coordinate of the starting point
    \param y1 second coordinate of the finishing point
    \param z0 third coordinate of the starting point
    \param z1 third coordinate of the finishing point
    \param t0 fourth coordinate of the starting point
    \param t1 fourth coordinate of the finishing point
    \param fX subsampling factor in the first direction
    \param fY subsampling factor in the second direction
    \param fZ subsampling factor in the third direction
    \param fT subsampling factor in the fourth direction
  */ 
  extern fff_array fff_array_get_block(const fff_array* thisone, 
				       size_t x0, size_t x1, size_t fX,
				       size_t y0, size_t y1, size_t fY,
				       size_t z0, size_t z1, size_t fZ,
				       size_t t0, size_t t1, size_t fT);  
  
  extern void fff_array_extrema (double* min, double* max, const fff_array* thisone);

  extern void fff_array_copy(fff_array* ares, const fff_array* asrc);

  extern void fff_array_compress(fff_array* ares, const fff_array* asrc,
				 double r0, double s0, 
				 double r1, double s1); 

  extern void fff_array_add (fff_array * x, const fff_array * y);
  extern void fff_array_sub (fff_array * x, const fff_array * y);
  extern void fff_array_div (fff_array * x, const fff_array * y);
  extern void fff_array_mul (fff_array * x, const fff_array * y);





  /* 
     Convert image values to [0,clamp-1]; typically clamp = 256.
     Possibly modify the dynamic range if the input value is
     overestimated.  For instance, the reconstructed MRI signal is
     generally encoded in 12 bits (values ranging from 0 to
     4095). Therefore, this operation may result in a loss of
     information.
  */
  extern void fff_array_clamp(fff_array* ares, const fff_array* asrc, double th, int* clamp); 

  extern fff_array_iterator fff_array_iterator_init(const fff_array* array); 
  extern fff_array_iterator fff_array_iterator_init_skip_axis(const fff_array* array, int axis); 
  
  /*  extern void fff_array_iterator_update(fff_array_iterator* thisone); */
  extern void fff_array_iterate_vector_function(fff_array* array, int axis, 
						void(*func)(fff_vector*, void*), void* par); 

#ifdef __cplusplus
}
#endif

#endif
