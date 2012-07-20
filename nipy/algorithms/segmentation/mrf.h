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
		      const PyArrayObject* U, 
		      int ngb_size,
		      double beta);

  extern PyArrayObject* make_edges(const PyArrayObject* mask,
				   int ngb_size);

  extern double interaction_energy(PyArrayObject* ppm, 
				   const PyArrayObject* XYZ,
				   const PyArrayObject* U,
				   int ngb_size);

#ifdef __cplusplus
}
#endif

#endif
