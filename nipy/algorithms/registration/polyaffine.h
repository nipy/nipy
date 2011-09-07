#ifndef POLYAFFINE
#define POLYAFFINE

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include <numpy/arrayobject.h>

  extern void polyaffine_import_array(void);

  extern void apply_polyaffine(PyArrayObject* XYZ, 
			       const PyArrayObject* Centers, 
			       const PyArrayObject* Affines, 
			       const PyArrayObject* Sigma); 


#ifdef __cplusplus
}
#endif

#endif
