#ifndef QUANTILE
#define QUANTILE

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include <numpy/arrayobject.h>

  extern double quantile(double* data,
			 npy_intp size,
			 npy_intp stride,
			 double r,
			 int interp);


#ifdef __cplusplus
}
#endif

#endif
