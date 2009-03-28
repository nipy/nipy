/* Special functions for FFF.
 * Author: Gael Varoquaux (implemented from canonical sources: 
 *		log gammma: algorithm as described in numerical recipes
 *		psi : algorithm as described in Applied Statistics,
 *		      Volume 25, Number 3, 1976, pages 315-317.
 *
 * License: BSD
 */

#include "fff_specfun.h"
#include <math.h>

double fff_gamln(double x)
{
  /* Log Gamma.
   *
   * INPUT: x > 0
   */
  double coeff[] = { 76.18009172947146,
                  -86.50532032941677,
                   24.01409824083091,
                  -1.231739572450155,
                   .1208650973866179e-2,
                  -.5395239384953e-5 };
  const double stp = 2.5066282746310005;
  double y = x;
  double sum = 1.000000000190015;
  double out ;
  int i;
  for(i=0; i<6; i++)
  {
    y += 1;
    sum += coeff[i]/y;
  }
  out = x + 5.5;
  out = (x+0.5) * log(out) - out;
  return out + log(stp*sum/x);
}


double fff_psi(double x)
{
  /* psi: d gamln(x)/dx
  *
  * INPUT:  x > 0
  */
  double c = 8.5;
  double d1 = -0.5772156649;
  double r;
  double s = 0.00001;
  double s3 = 0.08333333333;
  double s4 = 0.0083333333333;
  double s5 = 0.003968253968;
  double out;
  double y;
  /* XXX: What if x < 0 ? */
  y = x;
  out = 0.0;
  /* Use approximation if argument <= s */
  if (y<= s)
  {
    out = d1 - 1.0 / y;
    return out;
  }
  /* Reduce to psi(x + n) where (x + n) >= c */
  while (y<c)
  {
    out = out - 1.0/y;
    y = y + 1.0;
  }
  /* Use Stirling's expansion if argument > c */
  r = 1.0 / y;
  out += log (y) - 0.5*r;
  r = r*r;
  out += -r*(s3 - r * ( s4 - r*s5));
  return out;
}



