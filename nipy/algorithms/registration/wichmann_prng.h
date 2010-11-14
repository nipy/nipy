#ifndef WICHMANN_PRNG
#define WICHMANN_PRNG

#ifdef __cplusplus
extern "C" {
#endif
  
  /*
    B.A. Wichmann, I.D. Hill, Generating good pseudo-random numbers,
    Computational Statistics & Data Analysis, Volume 51, Issue 3, 1
    December 2006, Pages 1614-1622, ISSN 0167-9473, DOI:
    10.1016/j.csda.2006.05.019.
   */

  typedef struct {
    int ix; 
    int iy; 
    int iz; 
    int it; 
  } prng_state;

  extern void prng_seed(int seed, prng_state* rng);
  extern double prng_double(prng_state* prng); 

#ifdef __cplusplus
}
#endif

#endif
