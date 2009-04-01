#include "fff_base.h"


unsigned int fff_nbytes(fff_datatype type)
{
  unsigned int nbytes; 
  
  switch(type) {
  case FFF_UCHAR:
    nbytes = (unsigned int)sizeof(unsigned char); 
    break;
  case FFF_SCHAR:
    nbytes = (unsigned int)sizeof(signed char); 
    break;
  case FFF_USHORT: 
    nbytes = (unsigned int)sizeof(unsigned short); 
    break;
  case FFF_SSHORT: 
    nbytes = (unsigned int)sizeof(signed short); 
    break;
  case FFF_UINT: 
    nbytes = (unsigned int)sizeof(unsigned int); 
    break;
  case FFF_INT: 
    nbytes = (unsigned int)sizeof(int); 
    break;
  case FFF_ULONG: 
    nbytes = (unsigned int)sizeof(unsigned long); 
    break;
  case FFF_LONG:
    nbytes = (unsigned int)sizeof(long); 
    break;
  case FFF_FLOAT: 
    nbytes = (unsigned int)sizeof(float); 
    break;
  case FFF_DOUBLE: 
    nbytes = (unsigned int)sizeof(double); 
    break;
  default: 
    nbytes = 0;  
    break; 
  }
  return nbytes; 
}




int fff_is_integer(fff_datatype type)
{
  int ok = 0; 

  switch (type) {

  default:
    break; 

  case FFF_UCHAR:
  case FFF_SCHAR:
  case FFF_USHORT:
  case FFF_SSHORT:
  case FFF_UINT:
  case FFF_INT:
  case FFF_ULONG:
  case FFF_LONG:
    ok = 1; 
    break;
  
  }

  return ok; 
}


fff_datatype fff_get_datatype( unsigned int sizeType, 
			       unsigned int integerType, 
			       unsigned int signedType )
{
  fff_datatype type = FFF_UNKNOWN_TYPE; 

  /* Case: integer type */ 
  if ( integerType ) {

    if ( signedType ) {
      if ( sizeType == sizeof(signed char) ) 
	type = FFF_SCHAR;
      else if ( sizeType == sizeof(signed short) )
	type = FFF_SSHORT;
      else if ( sizeType == sizeof(int) )
	type = FFF_INT;
      else if ( sizeType == sizeof(signed long int) )
	type = FFF_LONG;
    }
    else {
      if ( sizeType == sizeof(unsigned char) ) 
	type = FFF_UCHAR;
      else if ( sizeType == sizeof(unsigned short) )
	type = FFF_USHORT;
      else if ( sizeType == sizeof(unsigned int) )
	type = FFF_UINT;
      else if ( sizeType == sizeof(unsigned long int) )
	type = FFF_ULONG;
    }
  
  }
  
  /* Case: floating type */ 
  else {
    if ( sizeType == sizeof(float) ) 
      type = FFF_FLOAT;
    else if ( sizeType == sizeof(double) )
      type = FFF_DOUBLE;
  }

  return type; 

}

