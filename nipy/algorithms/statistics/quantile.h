#ifndef QUANTILE
#define QUANTILE

#ifdef __cplusplus
extern "C" {
#endif

  extern double quantile(double* data,
			 unsigned long size,
			 unsigned long stride,
			 double r,
			 int interp);


#ifdef __cplusplus
}
#endif

#endif
