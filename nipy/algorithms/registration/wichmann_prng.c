#include "wichmann_prng.h"

#include <stdlib.h>

/*
  Assumption to be verified: 
  ix, iy, iz, it should be set to values between 1 and 400000
 */
void prng_seed(int seed, prng_state* rng)
{
  double r, rmax=(double)RAND_MAX; 
  int imax = 400000; 
  srand(seed); 

  r = (double)rand()/rmax;
  rng->ix = (int)(imax*r);  
  r = (double)rand()/rmax;
  rng->iy = (int)(imax*r);  
  r = (double)rand()/rmax;
  rng->iz = (int)(imax*r);  
  r = (double)rand()/rmax;
  rng->it = (int)(imax*r);  

  return; 
}


double prng_double(prng_state* rng)
{
  double W; 

  rng->ix = 11600 * (rng->ix % 185127) - 10379 * (rng->ix / 185127);
  rng->iy = 47003 * (rng->iy %  45688) - 10479 * (rng->iy /  45688);
  rng->iz = 23000 * (rng->iz %  93368) - 19423 * (rng->iz /  93368);
  rng->it = 33000 * (rng->it %  65075) -  8123 * (rng->it /  65075);
    
  if (rng->ix < 0)
    rng->ix = rng->ix + 2147483579;
  if (rng->iy < 0)
    rng->iy = rng->iy + 2147483543;
  if (rng->iz < 0)
    rng->iz = rng->iz + 2147483423;
  if (rng->it < 0)
    rng->it = rng->it + 2147483123;
  
  W = rng->ix/2147483579. + rng->iy/2147483543. 
    + rng->iz/2147483423. + rng->it/2147483123.;
    
  return W - (int)W;
}

