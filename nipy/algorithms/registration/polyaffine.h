#ifndef POLYAFFINE
#define POLYAFFINE

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>

/*
 * Use extension numpy symbol table
 */
#define NO_IMPORT_ARRAY
#include "_registration.h"

#include <numpy/arrayobject.h>

  extern void apply_polyaffine(PyArrayObject* XYZ, 
			       const PyArrayObject* Centers, 
			       const PyArrayObject* Affines, 
			       const PyArrayObject* Sigma); 


#ifdef __cplusplus
}
#endif

#endif
