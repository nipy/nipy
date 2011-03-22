#ifndef MRF
#define MRF

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include <numpy/arrayobject.h>

  extern void mrf_import_array(void);

  extern void ve_step(PyArrayObject* ppm, 
		      const PyArrayObject* ref,
		      const PyArrayObject* XYZ, 
		      const PyArrayObject* mix, 
		      double beta,
		      int copy,
		      int hard);

  extern double interaction_energy(PyArrayObject* ppm, 
				   const PyArrayObject* XYZ, 
				   const PyArrayObject* mix); 


#ifdef __cplusplus
}
#endif

#endif
