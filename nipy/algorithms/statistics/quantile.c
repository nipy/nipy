#include "quantile.h"

#include <stdio.h>
#include <math.h>

#ifdef INFINITY
#define POSINF INFINITY
#else 
#define POSINF HUGE_VAL
#endif

#define UNSIGNED_FLOOR(a) ( (int)(a) )
#define UNSIGNED_CEIL(a) ( ( (int)(a)-a )!=0.0 ? (int)(a+1) : (int)(a) )
#define SWAP(a, b)  {tmp=(a); (a)=(b); (b)=tmp;}


/* Declaration of static functions */ 
static double _pth_element(double* x,
			   npy_intp p, 
			   npy_intp stride,
			   npy_intp size); 
static void _pth_interval(double* am,
			  double* aM,
			  double* x,
			  npy_intp p,
			  npy_intp stride,
			  npy_intp size); 

/* 
   Quantile. 

   Given a sample x, this function computes a value q so that the
   number of sample values that are greater or equal to q is smaller
   or equal to (1-r) * sample size. 
*/ 
double quantile(double* data,
		npy_intp size,
		npy_intp stride,
		double r,
		int interp)
{
  double m, pp; 
  npy_intp p;

  if ((r<0) || (r>1)){
    fprintf(stderr, "Ratio must be in [0,1], returning zero"); 
    return 0.0; 
  }

  if (size == 1) 
    return data[0];   
  
  /* Find the smallest index p so that p >= r * size */ 
  if (!interp) {
    pp = r * size; 
    p = UNSIGNED_CEIL(pp); 
    if (p == size) 
      return POSINF; 
    m = _pth_element(data, p, stride, size);
  }
  else {
    double wm, wM; 
    pp = r * (size-1); 
    p = UNSIGNED_FLOOR(pp);
    wM = pp - (double)p;
    wm = 1.0 - wM; 
    if (wM <= 0) 
      m = _pth_element(data, p, stride, size);
    else { 
      double am, aM; 
      _pth_interval(&am, &aM, data, p, stride, size);
      m = wm*am + wM*aM; 
    }
  }
  
  return m; 
}


/*** STATIC FUNCTIONS ***/ 
/* BEWARE: the input array x gets modified! */ 

/*
  Pick up the sample value a so that:
  (p+1) sample values are <= a AND the remaining sample values are >= a 

*/ 


static double _pth_element(double* x,
			   npy_intp p,
			   npy_intp stride,
			   npy_intp n)
{
  double a, tmp;  
  double *bufl, *bufr;
  npy_intp i, j, il, jr, stop1, stop2;
  int same_extremities;
   
  stop1 = 0; 
  il = 0; 
  jr = n-1;
  while (stop1 == 0) {
    
    same_extremities = 0; 
    bufl = x + stride*il; 
    bufr = x + stride*jr; 
    if (*bufl > *bufr)
      SWAP(*bufl, *bufr)
    else if (*bufl == *bufr) 
      same_extremities = 1;
    a = *bufl; 
    
    if (il == jr) 
      return a;
    bufl += stride; 
    i = il + 1; 
    j = jr; 
    
    stop2 = 0; 
    while (stop2 == 0) {
      while (*bufl < a) {
	i ++;
	bufl += stride;
      }
      while (*bufr > a) {
	j --; 
	bufr -= stride; 
      }
      if (j <= i) 
	stop2 = 1; 
      else {
	SWAP(*bufl, *bufr)
	j --; bufr -= stride;
	i ++; bufl += stride; 
      } 
      
      /* Avoids infinite loops in samples with redundant values. 
         This situation can only occur with i == j */ 
      if ((same_extremities) && (j==jr)) {
	j --; 
	bufr -= stride; 
	SWAP(x[il*stride], *bufr) 
	stop2 = 1; 
      }
    }   
    
    /* At this point, we know that il <= j <= i; moreover: 
         if k <= j, x(j) <= a and if k > j, x(j) >= a 
         if k < i, x(i) <= a and if k >= i, x(i) >= a 
      
       We hence have: (j+1) values <= a and the remaining (n-j-1) >= a 
                        i values <= a and the remaining (n-i) >= a                        
    */
  
    if (j > p) 
      jr = j;
    else if (j < p)
      il = i;
    else /* j == p */
      stop1 = 1;

  }

  return a; 
}


/* BEWARE: the input array x gets modified! */ 
static void _pth_interval(double* am,
			  double* aM, 
			  double* x,
			  npy_intp p,
			  npy_intp stride,
			  npy_intp n)
{
  double a, tmp;
  double *bufl, *bufr; 
  npy_intp i, j, il, jr, stop1, stop2, stop3;
  npy_intp pp = p+1; 
  int same_extremities = 0; 

  *am = 0.0; 
  *aM = 0.0; 
  stop1 = 0; 
  stop2 = 0; 
  il = 0; 
  jr = n-1; 
  while ((stop1 == 0) || (stop2 == 0)) {

    same_extremities = 0; 
    bufl = x + stride*il; 
    bufr = x + stride*jr; 
    if (*bufl > *bufr)
      SWAP(*bufl, *bufr) 
    else if (*bufl == *bufr) 
      same_extremities = 1;
    a = *bufl;
    
    if (il == jr) { 
      *am=a; 
      *aM=a; 
      return;
    } 
    
    bufl += stride; 
    i = il + 1; 
    j = jr; 

    stop3 = 0; 
    while (stop3 == 0) {

      while (*bufl < a) {
	i ++;
	bufl += stride;
      }
      while (*bufr > a) {
	j --; 
	bufr -= stride; 
      }
      if (j <= i)  
	stop3 = 1; 
      else {
	SWAP(*bufl, *bufr)
	j --; bufr -= stride;
	i ++; bufl += stride; 
      } 

      /* Avoids infinite loops in samples with redundant values */ 
      if ((same_extremities) && (j==jr)) {
	j --; 
	bufr -= stride; 
	SWAP(x[il*stride], *bufr) 
        stop3 = 1; 
      }

    }

    /* At this point, we know that there are (j+1) datapoints <=a
       including a itself, and another (n-j-1) datapoints >=a */
    if (j > pp) 
      jr = j;
    else if (j < p) 
      il = i;
    /* Case: found percentile at p */
    else if (j == p) { 
      il = i; 
      *am = a; 
      stop1 = 1; 
    }
    /* Case: found percentile at (p+1), ie j==(p+1) */ 
    else { 
      jr = j; 
      *aM = a;
      stop2 = 1;
    }

  }

  return;  
}

