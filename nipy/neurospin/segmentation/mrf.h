#ifndef MRF
#define MRF

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include <numpy/arrayobject.h>

  extern void mrf_mport_array(void);

  extern void ve_step(PyArrayObject* ppm, 
		      const PyArrayObject* ref,
		      const PyArrayObject* XYZ, 
		      double beta,
		      int copy,
		      int hard);

  extern double concensus(PyArrayObject* ppm, 
			  const PyArrayObject* XYZ); 


#ifdef __cplusplus
}
#endif

#endif
