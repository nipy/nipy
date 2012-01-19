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
		      int ngb_size,
		      double beta,
		      int copy,
		      int mtype);

  extern double interaction_energy(PyArrayObject* ppm, 
				   const PyArrayObject* XYZ,
				   int ngb_size); 

  extern void gen_ve_step(PyArrayObject* ppm, 
			  const PyArrayObject* ref,
			  const PyArrayObject* XYZ, 
			  const PyArrayObject* U, 
			  int ngb_size,
			  double beta);


#ifdef __cplusplus
}
#endif

#endif
