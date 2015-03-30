/*!
  \file fff_base.h
  \brief Basic fff macros and error handling functions 
  \author Alexis Roche
  \date 2003-2008

*/

#ifndef FFF_BASE
#define FFF_BASE
 
#ifdef __cplusplus
extern "C" {
#endif


#include<math.h>
#include <stdio.h>


#ifdef INFINITY
#define FFF_POSINF INFINITY
#define FFF_NEGINF (-INFINITY)
#else 
#define FFF_POSINF HUGE_VAL
#define FFF_NEGINF (-HUGE_VAL)
#endif

#ifdef NAN
#define FFF_NAN NAN
#else
#define FFF_NAN (FFF_POSINF/FFF_POSINF)
#endif

#ifdef NO_APPEND_FORTRAN
# define FFF_FNAME(x) x
#else
# define FFF_FNAME(x) x##_
#endif 


  /*! 
    Displays an error message with associated error code.
  */
#define FFF_ERROR(message, errcode)					\
  {									\
    fprintf(stderr, "Unhandled error: %s (errcode %i)\n", message, errcode); \
    fprintf(stderr, " in file %s, line %d, function %s\n",  __FILE__, __LINE__, __FUNCTION__); \
  }									\
    
  /*! 
    Displays a warning message.
  */
#define FFF_WARNING(message)						\
  {									\
    fprintf(stderr, "Warning: %s\n", message);				\
    fprintf(stderr, " in file %s, line %d, function %s\n",  __FILE__, __LINE__, __FUNCTION__); \
  }									\
   

  /*! 
    Displays a debug message.
  */
#define FFF_DEBUG(message)						\
  {									\
    fprintf(stderr, "DEBUG: %s\n", message);				\
    fprintf(stderr, " in file %s, line %d, function %s\n",  __FILE__, __LINE__, __FUNCTION__); \
  }									\
    
  

  /*!
    Rounds \a a to the nearest smaller integer 
    \bug Compilator-dependent?
  */
#define FFF_FLOOR(a)((a)>0.0 ? (int)(a):(((int)(a)-a)!= 0.0 ? (int)(a)-1 : (int)(a)))  
  /*!
    Rounds \a a to the nearest integer (either smaller or bigger) 
  */
#define FFF_ROUND(a)(FFF_FLOOR(a+0.5))
  /*!
    Rounds \a a to the nearest bigger integer 
  */
#define FFF_CEIL(a)(-(FFF_FLOOR(-(a))))
  /*!
    Rounds \a a to the nearest smaller integer, assuming \a a is non-negative 
    \bug Compilator-dependent?
  */
#define FFF_UNSIGNED_FLOOR(a) ( (int)(a) )
  /*!
    Rounds \a a to the nearest integer, assuming \a a is non-negative 
  */
#define FFF_UNSIGNED_ROUND(a) ( (int)(a+0.5) )
  /*!
    Rounds \a a to the nearest bigger integer, assuming \a a is non-negative 
  */
#define FFF_UNSIGNED_CEIL(a) ( ( (int)(a)-a )!=0.0 ? (int)(a+1) : (int)(a) )
  /*!
    Returns 1 if \a a is positive, -1 if \a a is negative, 0 if \a a equals zero

    Note that this macro differs from \a GSL_SIGN which returns +1 if \a a==0
  */
#define FFF_SIGN(a)( (a)>0.0 ? 1 : ( (a)<0.0 ? -1 : 0 ) )
  /*!
    Computes the absolute value of \a a
  */
#define FFF_ABS(a) ( (a) > 0.0 ? (a) : (-(a)) )
  /*!
    Computes \f$ a^2 \f$
  */
#define FFF_SQR(a) ( (a)*(a) )
  /*!
    Computes \f$ a^3 \f$
  */
#define FFF_CUBE(a) ( (a)*(a)*(a) )
  /*!
    Computes \f$ a modulo, b ie the remainder after division of a by b  \f$
  */
#define FFF_REM(a, b) ( (int)(a)%(int)(b) )
  /*!
    Computes the minimum of \a a and \a b
  */
#define FFF_MIN(a,b) ( (a) < (b) ? (a) : (b) ) 
  /*!
    Computes the maximum of \a a and \a b
  */
#define FFF_MAX(a,b) ( (a) > (b) ? (a) : (b) ) 
  /*!
    Low threshold a value to avoid vanishing 
  */
#define FFF_TINY 1e-50
#define FFF_ENSURE_POSITIVE(a) ( (a) > FFF_TINY ? (a) : FFF_TINY )

#define FFF_IS_ODD(n) ((n) & 1)


  /*!
    \typedef fff_datatype
    \brief Data encoding types
  */
  typedef enum {
    FFF_UNKNOWN_TYPE = -1,  /*!< unknown type */
    FFF_UCHAR = 0,          /*!< unsigned char */
    FFF_SCHAR = 1,          /*!< signed char */
    FFF_USHORT = 2,         /*!< unsigned short */
    FFF_SSHORT = 3,         /*!< signed short */
    FFF_UINT = 4,           /*!< unsigned int */
    FFF_INT = 5,            /*!< (signed) int */
    FFF_ULONG = 6,          /*!< unsigned long int */
    FFF_LONG = 7,           /*!< (signed) long int */
    FFF_FLOAT = 8,          /*!< float */
    FFF_DOUBLE = 9          /*!< double */
  } fff_datatype;


  /*!
    \brief Return the byte length of a given data type
    \param type input data type
  */
  extern unsigned int fff_nbytes(fff_datatype type); 

  /*!
    \brief Return 1 if data type is integer, 0 otherwise
    \param type input data type
  */
  extern int fff_is_integer(fff_datatype type);

  /*!
    \brief Return the data type that matches given features
    \param sizeType size in bytes
    \param integerType if zero, a floating-point type (\c float or \c double) is assumed
    \param signedType for integer types, tells whether the type is signed or not
  */
  extern fff_datatype fff_get_datatype( unsigned int sizeType, 
					unsigned int integerType, 
					unsigned int signedType ); 



#ifdef __cplusplus
}
#endif
 
#endif  
