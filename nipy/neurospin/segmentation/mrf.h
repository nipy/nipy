#ifndef MRF
#define MRF

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include <numpy/arrayobject.h>

  extern void mrf_mport_array(void);

  extern void smooth_ppm(PyArrayObject* ppm, 
			 const PyArrayObject* lik,
			 const PyArrayObject* XYZ, 
			 double beta,
                         int copy,
                         int hard);

    

#ifdef __cplusplus
}
#endif

#endif
