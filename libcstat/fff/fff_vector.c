#include "fff_base.h"
#include "fff_vector.h"
#include "fff_array.h"

#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>

/* Declaration of static functions */ 
static double _fff_pth_element(double* x, size_t p, size_t stride, size_t size); 
static void _fff_pth_interval(double* am, double* aM, 
			       double* x, size_t p, size_t stride, size_t size); 


/* Constructor */ 
fff_vector* fff_vector_new(size_t size)
{
  fff_vector* thisone;

  thisone = (fff_vector*)calloc(1, sizeof(fff_vector)); 
  if (thisone == NULL) { 
    FFF_ERROR("Allocation failed", ENOMEM); 
    return NULL; 
  }

  thisone->data = (double*)calloc(size, sizeof(double)); 
  if (thisone->data == NULL) 
    FFF_ERROR("Allocation failed", ENOMEM); 

  thisone->size = size; 
  thisone->stride = 1; 
  thisone->owner = 1; 
  
  return thisone; 
}

/* Destructor */ 
void fff_vector_delete(fff_vector* thisone)
{
  if (thisone->owner) 
    if (thisone->data != NULL) 
      free(thisone->data); 
  free(thisone); 

  return; 
}

/* View */ 
fff_vector fff_vector_view(const double* data, size_t size, size_t stride)
{
  fff_vector x; 

  x.size = size; 
  x.stride = stride; 
  x.owner = 0; 
  x.data = (double*)data; 

  return x; 
}




#define CHECK_SIZE(x,y)						\
  if ((x->size) != (y->size)) FFF_ERROR("Vectors have different sizes", EDOM)


/* Vector copy. If both vectors are contiguous in memory, we use
   memcpy, otherwise we perform a loop */ 
void fff_vector_memcpy(fff_vector* x, const fff_vector* y)
{
  CHECK_SIZE(x, y); 

  if ((x->stride == 1) && (y->stride == 1))
    memcpy((void*)x->data, (void*)y->data, x->size*sizeof(double));
  else {
    size_t i; 
    double *bx, *by; 
    for(i=0, bx=x->data, by=y->data; i<x->size; i++, bx+=x->stride, by+=y->stride)
      *bx = *by; 
  }
  
  return; 
}


/* Copy buffer with arbitrary type */
void fff_vector_fetch(fff_vector* x, const void* data, fff_datatype datatype, size_t stride)
{
  fff_array a = fff_array_view1d(datatype, (void*)data, x->size, stride);
  fff_array b =  fff_array_view1d(FFF_DOUBLE, x->data, x->size, x->stride);

  fff_array_copy(&b, &a); 
  
  return;
}



/* Get an element */
double fff_vector_get (const fff_vector * x, size_t i)
{
  return(x->data[ i * x->stride ]); 
}

/* Set an element */ 
void fff_vector_set (fff_vector * x, size_t i, double a)
{
  x->data[ i * x->stride ] = a; 
  return; 
}

/* Set all elements */ 
void fff_vector_set_all (fff_vector * x, double a)
{
  size_t i; 
  double *buf; 
  for(i=0, buf=x->data; i<x->size; i++, buf+=x->stride)
    *buf = a; 
  return; 
}

/* Add two vectors */ 
void fff_vector_add (fff_vector * x, const fff_vector * y)
{
  size_t i; 
  double *bx, *by; 
  CHECK_SIZE(x, y); 
  for(i=0, bx=x->data, by=y->data; i<x->size; i++, bx+=x->stride, by+=y->stride)
    *bx += *by; 
  return; 
}

/* Compute: x = x - y */ 
void fff_vector_sub (fff_vector * x, const fff_vector * y)
{
  size_t i; 
  double *bx, *by; 
  CHECK_SIZE(x, y); 
  for(i=0, bx=x->data, by=y->data; i<x->size; i++, bx+=x->stride, by+=y->stride)
    *bx -= *by; 
  return; 
} 

/* Element-wise product */
void fff_vector_mul (fff_vector * x, const fff_vector * y)
{
  size_t i; 
  double *bx, *by; 
  CHECK_SIZE(x, y); 
  for(i=0, bx=x->data, by=y->data; i<x->size; i++, bx+=x->stride, by+=y->stride)
    *bx *= *by; 
  return; 
}

/* Element-wise division */ 
void fff_vector_div (fff_vector * x, const fff_vector * y)
{
  size_t i; 
  double *bx, *by; 
  CHECK_SIZE(x, y); 
  for(i=0, bx=x->data, by=y->data; i<x->size; i++, bx+=x->stride, by+=y->stride)
    *bx /= *by; 
  return; 
}

/* Scale by a constant */ 
void fff_vector_scale (fff_vector * x, double a)
{
  size_t i; 
  double *bx; 
  for(i=0, bx=x->data; i<x->size; i++, bx+=x->stride)
    *bx *= a; 
  return; 
}

/* Add a constant */ 
void fff_vector_add_constant (fff_vector * x, double a)
{
  size_t i; 
  double *bx; 
  for(i=0, bx=x->data; i<x->size; i++, bx+=x->stride)
    *bx += a; 
  return; 
}


/* Sum up elements */ 
long double fff_vector_sum(const fff_vector* x)
{
  long double sum = 0.0; 
  double* buf = x->data; 
  size_t i; 

  for(i=0; i<x->size; i++, buf+=x->stride)
    sum += *buf; 

  return sum; 
} 

/* Mean */ 
double fff_vector_mean(const fff_vector* x) {
  return((double)(fff_vector_sum(x) / (double)x->size)); 
}

/* SSD 

We use Konig formula: 

SUM[(x-a)^2] = SUM[(x-m)^2] + n*(a-m)^2
where m is the mean. 

*/ 
long double fff_vector_ssd(const fff_vector* x, double* m, int fixed_offset)
{
  long double ssd = 0.0;
  long double sum = 0.0; 
  long double n = (long double)x->size; 
  double aux; 
  double* buf = x->data; 
  size_t i; 

  for(i=0; i<x->size; i++, buf+=x->stride) {
    aux = *buf; 
    sum += aux;
    ssd += FFF_SQR(aux);  
  }

  sum /= n; 
  if (fixed_offset) {
    aux = *m - sum; 
    ssd += n * (FFF_SQR(aux) - FFF_SQR(sum)); 
  }
  else{
    *m = sum; 
    ssd -= n * FFF_SQR(sum); 
  }
  
  return ssd; 
}


long double fff_vector_wsum(const fff_vector* x, const fff_vector* w, long double* sumw)
{
  long double wsum=0.0, aux=0.0; 
  double *bufx=x->data, *bufw=w->data; 
  size_t i; 
  CHECK_SIZE(x, w); 
  for(i=0; i<x->size; i++, bufx+=x->stride, bufw+=w->stride) {
    wsum += (*bufw) * (*bufx); 
    aux += *bufw; 
  }
  *sumw = aux; 
  return wsum; 
}

long double fff_vector_sad(const fff_vector* x, double m)
{
  long double sad=0.0;
  double aux; 
  double *buf=x->data;
  size_t i; 
  for(i=0; i<x->size; i++, buf+=x->stride) {
    aux = *buf-m; 
    sad += FFF_ABS(aux);
  }
  return sad;
} 
  



/* Median (modify input vector) */ 
double fff_vector_median(fff_vector* x)
{
  double m;
  double* data = x->data; 
  size_t stride = x->stride, size = x->size; 
  
  if (FFF_IS_ODD(size)) 
    m = _fff_pth_element(data, size>>1, stride, size); 
  
  else{ 
    double mm;
    _fff_pth_interval(&m, &mm, data, (size>>1)-1, stride, size); 
    m = .5*(m+mm);
  }
  
  return m; 
}


/* 
   Quantile. 

   Given a sample x, this function computes a value q so that the
   number of sample values that are greater or equal to q is smaller
   or equal to (1-r) * sample size. 
*/ 
double fff_vector_quantile(fff_vector* x, double r, int interp)
{
  double m, pp; 
  double* data = x->data; 
  size_t p, stride = x->stride, size = x->size; 

  if ((r<0) || (r>1)){
    FFF_WARNING("Ratio must be in [0,1], returning zero"); 
    return 0.0; 
  }

  if (size == 1) 
    return data[0];   
  
  /* Find the smallest index p so that p >= r * size */ 
  if (!interp) {
    pp = r * size; 
    p = FFF_UNSIGNED_CEIL(pp); 
    if (p == size) 
      return FFF_POSINF; 
    m = _fff_pth_element(data, p, stride, size);
  }
  else {
    double wm, wM; 
    pp = r * (size-1); 
    p = FFF_UNSIGNED_FLOOR(pp);
    wM = pp - (double)p;
    wm = 1.0 - wM; 
    if (wM <= 0) 
      m = _fff_pth_element(data, p, stride, size);
    else { 
      double am, aM; 
      _fff_pth_interval(&am, &aM, data, p, stride, size);
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

#define SWAP(a, b)  {tmp=(a); (a)=(b); (b)=tmp;}
static double _fff_pth_element(double* x, size_t p, size_t stride, size_t n)
{
  double a, tmp;  
  double *bufl, *bufr;
  size_t i, j, il, jr, stop1, stop2;
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
static void _fff_pth_interval(double* am, double* aM, 
			       double* x, size_t p, size_t stride, size_t n)
{
  double a, tmp;
  double *bufl, *bufr; 
  size_t i, j, il, jr, stop1, stop2, stop3;
  size_t pp = p+1; 
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

/*
  Sort x  by ascending order and reorder w accordingly. 
*/

double fff_vector_wmedian_from_sorted_data (const fff_vector* x_sorted,
					    const fff_vector* w)
{
  size_t i; 
  double mu, sumW, WW, WW_prev, xx, xx_prev, ww;  
  double *bxx, *bww; 

  /* Compute the sum of weights */ 
  sumW = (double) fff_vector_sum(w);
  if (sumW <= 0.0) 
    return FFF_NAN; 

  /* Find the smallest index such that the cumulative density > 0.5 */ 
  i = 0; 
  xx = FFF_NEGINF;
  WW = 0.0;  
  bxx = x_sorted->data; 
  bww = w->data; 
  while (WW <= .5) {
    xx_prev = xx;
    WW_prev = WW;  
    xx = *bxx;
    ww = *bww / sumW;
    WW += ww; 
    i ++; 
    bxx += x_sorted->stride; 
    bww += w->stride; 
  }

  /* Linearly interpolated median */ 
  if (i == 1) 
    mu = xx; 
  else 
    mu = .5*(xx_prev+xx) + (.5-WW_prev)*(xx-xx_prev)/ww; 

  return mu; 
}

