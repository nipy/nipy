/*!

  \file cubic_spline.h
  \brief Cubic spline transformation and interpolation.
  \author Alexis Roche
  \date 2003

  Compute the cubic spline coefficients of regularly sampled signals
  and perform interpolation. The cubic spline transform is implemented
  from the recursive algorithm described in:

  M. Unser, "Splines : a perfect fit for signal/image processing ",
  IEEE Signal Processing Magazine, Nov. 1999.  Web page:
  http://bigwww.epfl.ch/publications/unser9902.html Please check the
  erratum.

*/


#ifndef CUBIC_SPLINE
#define CUBIC_SPLINE

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

  /*! 
    \brief Cubic spline basis function
    \param x input value 
  */
  extern double cubic_spline_basis(double x); 
  /*! 
    \brief Cubic spline transform of a one-dimensional signal 
    \param src input signal 
    \param res output signal (same size)
  */
  extern void cubic_spline_transform(PyArrayObject* res, const PyArrayObject* src);

  extern double cubic_spline_sample1d(double x, const PyArrayObject* coef, 
				      int mode); 
  extern double cubic_spline_sample2d(double x, double y, const PyArrayObject* coef,
				      int mode_x, int mode_y);
  extern double cubic_spline_sample3d(double x, double y, double z, const PyArrayObject* coef, 
				      int mode_x, int mode_y, int mode_z);
  extern double cubic_spline_sample4d(double x, double y, double z, double t, const PyArrayObject* coef,
				      int mode_x, int mode_y, int mode_z, int mode_t); 
  extern void cubic_spline_resample3d(PyArrayObject* im_resampled, const PyArrayObject* im, 
				      const double* Tvox, 
				      int mode_x, int mode_y, int mode_z);

    

#ifdef __cplusplus
}
#endif

#endif
