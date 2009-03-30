/*!
  \file fff_specfun.h
  \brief special functions needed by fff's C routines.
  \author Alexis Roche, Gael Varoquaux
  \date 2008, 2009
  \licence BSD

*/


#ifndef FFF_SPECFUN
#define FFF_SPECFUN
 
#ifdef __cplusplus
extern "C" {
#endif

  extern double fff_psi(double x);
  extern double fff_gamln(double x);

#ifdef __cplusplus
}
#endif
 
#endif  
