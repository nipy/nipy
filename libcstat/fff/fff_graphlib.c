#include "fff_graphlib.h"
#include "fff_field.h"
#include "fff_routines.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>

#define _SQRT2   1.41421356237309504880
#define _SQRT3   1.73205080756887719000


static long _fff_list_add(long *listn, double *listd, const long newn, const double newd, const long k);
static long _fff_list_move(long *listn, double *listd, const long newn, const double newd, const long k);
static double _fff_list_insertion(long *listn, double *listd, const long newn, const double newd, const long k);

static double _fff_g_euclidian(const fff_matrix* X, const long n1, const long n2);

static double _fff_cross_euclidian(const fff_matrix* X, const fff_matrix* Y, const long n1, const long n2);
static long _fff_graph_vect_neighb( fff_array *cindices, fff_array * neighb, fff_vector* weight, const fff_graph* G);
static void _fff_graph_preprocess_grid(long*u, long*MMx, long* MMxy, long* MMu, const long N, const long* xyz);
static void _fff_graph_preprocess_vgrid( long*u, long*MMx, long* MMxy, long* MMu,  const fff_array* xyz);
static void  _fff_sort_vector_index (fff_vector *dist, long* idx);

/******************************************************/
/**************** list handling ***********************/
/******************************************************/


static long  _fff_list_add( long *listn, double *listd,  const long newn, const double newd, const long k)
{  
  long i = k; 
  while (listd[i-1] > newd){
    listd[i] = listd[i-1];
    listn[i] = listn[i-1];
    i--;
    if (i<0){
      FFF_ERROR(" could not deal with newd ",EFAULT);
      /* return(1); */
    }
  } 
  listd[i] = newd;
  listn[i] = newn;
  return(0);
}

static long _fff_list_move( long *listn, double *listd,  const long newn, const double newd, const long k)
{ 
  long i = k-1;
  while (listn[i]!=newn) {
    i--;
    if (i<0){
	  FFF_ERROR("item not identified in the list",EFAULT);
    }
  }

  while (listd[i-1]>newd){
    listd[i] = listd[i-1];
    listn[i] = listn[i-1];
    i--;
    if (i<0){
       FFF_ERROR(" could not deal with newd ",EFAULT);
    }
  }
  listd[i] = newd;
  listn[i] = newn;
  return(0);
}

static double _fff_list_insertion(long *listn, double *listd, const long newn, const double newd, const long q)
{ 
  /* this is a suboptimal routine to insert a value into a sorted list */
  long i; 

  if (newd>listd[q-1])
	FFF_ERROR("insertion error ",EDOM);
  
  i = q-1;
  if (i>0)
    while (newd<listd[i-1]){  
      listd[i] = listd[i-1];
      listn[i] = listn[i-1];
      i--;
      if (i==0)break;
    }    
  listd[i] = newd;
  listn[i] =  newn;
  return(listd[q-1]);
}


static double _fff_g_euclidian(const fff_matrix* X, const long n1, const long n2)
{
  /* euclidian distance computation between two rows of the matrix*/
  long T = X->size2;
  long t;
  double dx;
  double dist = 0;

  for ( t=0 ; t<T ; t++){
    dx =  fff_matrix_get(X,n1,t)-fff_matrix_get(X,n2,t); 
    dist += dx*dx;
  } 
  return(sqrt(dist));
}

static double _fff_cross_euclidian(const fff_matrix* X,const fff_matrix* Y, const long n1, const long n2)
{
  /* caveat : dimension check */
  long T = X->size2;
  long t;
  double dx;
  double dist = 0;
  
  for ( t=0 ; t<T ; t++){
    dx = fff_matrix_get(X,n1,t)-fff_matrix_get(Y,n2,t); 
    dist += dx*dx;
  } 
  return(sqrt(dist));
}


/**********************************************************************
 ****************** sparse graph "contructor/destructor" **************
**********************************************************************/

fff_graph* fff_graph_new(const long v, const long e )
{
  long i;
  fff_graph * thisone;  

  /* Start with allocating the object */
  thisone = (fff_graph*) calloc( 1, sizeof(fff_graph) );
  
  /* Checks that the pointer has been allocated */
  if ( thisone == NULL) 
    return NULL; 

  /* Initialization */
  thisone->V = v;
  thisone->E = e;

  /* Allocate graph objects */

  thisone->eA = (long*) calloc( thisone->E,sizeof(long));
  thisone->eB = (long*) calloc( thisone->E,sizeof(long));
  thisone->eD = (double*) calloc( thisone->E,sizeof(double));

  /* Allocation test */ 
  if ( (thisone->eA == NULL) | (thisone->eB == NULL) | (thisone->eD == NULL)) {
    fff_graph_delete( thisone );
    return NULL;
  }

  for (i=0 ; i<thisone->E ; i++){
    thisone->eD[i] = 0;
    thisone->eA[i] = 0;
    thisone->eB[i] = 0;
  }
  
  return thisone;  
}

fff_graph* fff_graph_build(const long v, const long e, const long *A, const long* B, const double*D )
{
  fff_graph * thisone;  
   long i;

   /* contruction */
   thisone = fff_graph_new(v,e);
   if (thisone ==NULL) {
     FFF_WARNING(" Edge index is too high");
     return NULL;
   }

   /* Set the edge matrix */
   for (i=0 ; i<e ; i++){
     if (A[i]>v-1){
	   FFF_WARNING(" Edge index is too high");
	   return NULL;
     }
     if (B[i]>v-1){
	   FFF_WARNING(" Edge index is too high");
	   return NULL;
     }
     
     thisone->eA[i] = A[i];
     thisone->eB[i] = B[i];
     thisone->eD[i] = D[i];
   }
   return thisone;
}

extern fff_graph* fff_graph_build_safe(const long v, const long e, const fff_array *A, const fff_array* B, const fff_vector *D )
{
  fff_graph * thisone;  
  long i,ea,eb;

   /* check the dimensions */
  if (((A->dimX)!=e)|((B->dimX)!=e)|((D->size)!=e)){
	FFF_WARNING("inconsistant vector size \n");
	return NULL;
  }
  
  /* contruction */
  thisone = fff_graph_new(v,e);
  if (thisone == NULL) {
	FFF_WARNING("fff_graph_new did not alocate graph");
	return NULL;
  }

  /* Set the edge matrix */
  for (i=0 ; i<e ; i++){
	ea = fff_array_get1d(A,i);
	eb = fff_array_get1d(B,i);
	if (ea>v-1){
	  FFF_WARNING(" Edge index is too high");
	  return NULL;
	}
	if (eb>v-1){
	  FFF_WARNING(" Edge index is too high");
	  return NULL;
	}
	thisone->eA[i] = ea;
	thisone->eB[i] = eb;
	thisone->eD[i] = fff_vector_get(D,i);
  }
  return thisone;
}

void fff_graph_delete( fff_graph* thisone )
{

  if ( thisone != NULL ) {
    free(thisone->eA);
    free(thisone->eB);
    free(thisone->eD);
    free( thisone );
  }
  return;
}

extern void fff_graph_edit_safe(fff_array *A, fff_array* B, fff_vector *D, const fff_graph* thisone )
{
  long i; 
  size_t E = thisone->E;
  
  if (((A->dimX)!=E)|((B->dimX)!=E)|((D->size)!=E)) {
    FFF_ERROR("inconsistant vector size \n",EDOM);
  }
  else
    for (i=0 ; i<thisone->E ; i++){
      fff_array_set1d(A,i,thisone->eA[i]);
      fff_array_set1d(B,i,thisone->eB[i]);
      fff_vector_set(D,i,thisone->eD[i]);
    }
}

 /* This function recomputes the connectivity system of the graph in an efficient way:
  The edges are arraned in the following manner 
       origins =  [0..0 1..1 .. V-1..V-1]
       ends    =  [neignb[0].. neighb[E-1]]
       weight  =  [weights[0]..weights[E-1]]
       
       cindices codes for the origin vector: origin=i between cindices[i] and cindices[i+1]-1 
  */

static void _fff_sort_vector_index (fff_vector *dist, long* idx)
{
  /* Caveat: in this function dist is sorted */
  sort_ascending_and_get_permutation( dist->data, idx, dist->size );
 
}

/**********************************************************************
 *************************** Cartesian graphs (new version) ***********
**********************************************************************/
static void _fff_graph_preprocess_grid(long*u, long*MMx, long* MMxy, long* MMu, const long N, const long* xyz)
{
  /*   char* proc = "fff_graph_preprocess_grid"; */
  long i;
  long mx,my,mz,Mx,My,Mxy,Mu;

  /* Find minimal/maximal values of (x,y,z) */
  mx = xyz[0];
  my = xyz[N];
  mz = xyz[2*N];
  Mx = xyz[0];
  My = xyz[N];
  
  for (i=0 ; i<N ; i++){
    if (xyz[i]<mx) mx = xyz[i];
    if (xyz[i+N]<my) my = xyz[i+N];
    if (xyz[i+2*N]<mz) mz = xyz[i+2*N];   
    if (xyz[i]>Mx) Mx = xyz[i];
    if (xyz[i+N]>My) My = xyz[i+N];
  }
  Mx = Mx-mx+2;
  My = My-my+2;
  Mxy = Mx*My;
  Mu = 0;
  
  /* Code (x,y,z) by a scalar u*/
  for (i=0 ; i<N ; i++){
    u[i] = xyz[i]-mx + (xyz[i+N]-my)*Mx + (xyz[i+2*N]-mz)*Mxy;
    if (u[i]>Mu) Mu = u[i];
  }
  Mu = Mu+1;  
  
  *MMx = Mx;
  *MMxy = Mxy;
  *MMu = Mu;
}

long fff_graph_grid_six(fff_graph** G, const  long* xyz, const long N)
{
  /*   char* proc = "fff_graph_grid_six"; */

  fff_graph *thisone; 

  long E = 0;
  long i,j;
  long Mx,Mxy,Mu,ui;
  long *u, *A, *B, *invu; 
  double *D; 

  u = ( long*) calloc( N,sizeof(long));
  if (!u) return(0);
  A = ( long*) calloc( N*7,sizeof(long));
  if (!A) return(0);
  B = ( long*) calloc( N*7,sizeof(long));
  if (!B) return(0);
  D = ( double*) calloc( N*7,sizeof( double));
  if (!D) return(0);

  _fff_graph_preprocess_grid(u,&Mx,&Mxy, &Mu,N,xyz);
   
  /* find  invu such that  invu[u[i]]=i */
  invu = (long*) calloc( Mu,sizeof(long));
  if (!invu) return(0);
  for (i=0 ; i<(Mu) ; i++) invu[i]=-1;   
  for (i=0 ; i<N ; i++) invu[u[i]]=i;    
 
  /* Search for neighbours*/
  j=0;
  for (i=0 ; i<N ; i++){ 
    A[j] = i;
    B[j] = i;
    D[j] = 0;
    j++;   
    ui = u[i];
    if (ui+1 < Mu)
      if(invu[ui+1] > -1){
	A[j] = i;
	B[j] = invu[ui+1];
	D[j] = 1;
	j++;
      }
    if (ui > 0)
      if(invu[ui-1] > -1){
	A[j] = i;
	B[j] = invu[ui-1];
	D[j] = 1;
	j++;
      }
    if (ui+Mx < Mu)
      if(invu[ui+Mx]>-1){
	A[j] = i;
	B[j] = invu[ui+Mx];
	D[j] = 1;
	j++;
      }
    if (ui+1 > Mx)
      if(invu[ui-Mx]>-1){
	A[j] = i;
	B[j] = invu[ui-Mx];
	D[j] = 1;
	j++;
      }
    if (ui+Mxy<Mu)
      if(invu[ui + Mxy]>-1){
	A[j] = i;
	B[j] = invu[ui + Mxy];
	D[j] = 1;
	j++;
      }
    if (ui+1>Mxy)
      if(invu[ui-Mxy]>-1){
	A[j] = i;
	B[j] = invu[ui-Mxy];
	D[j] = 1;
	j++;
      }
  }
  E = j;

  thisone = fff_graph_build(N,E,A,B,D);
  
  *G = thisone;

  free(u);
  free(invu);
  free(A);
  free(B);
  free(D);
  return(E);
} 

long fff_graph_grid_eighteen(fff_graph** G, const long* xyz, const long N)
{
  /*   char* proc = "fff_graph_grid_eighteen"; */
  fff_graph* thisone;
  
  long E = 0;
  long i,j;
  long Mx,Mxy,Mu,ui;
  long *u, *A, *B, *invu; 
  double *D; 

  u = ( long*) calloc( N,sizeof(long));
  if (!u) {
    FFF_WARNING("calloc failed, graph to big?\n");
    return(0);
  }
  A = ( long*) calloc( N*19,sizeof(long));
  if (!A) {
    FFF_WARNING("calloc failed, graph to big?\n");
    return(0);
  }
  B = ( long*) calloc( N*19,sizeof(long));
  if (!B) {
    FFF_WARNING("calloc failed, graph to big?\n");
    return(0);
  }
  D = (double*) calloc( N*19,sizeof(double));
  if (!D) {
    FFF_WARNING("calloc failed, graph to big?\n");
    return(0);
  }

  _fff_graph_preprocess_grid(u,&Mx,&Mxy, &Mu,N,xyz);
 
  /* find  invu such that  invu[u[i]]=i */
  invu = (long*) calloc( Mu,sizeof(long));
  if (!invu) {
    FFF_WARNING("calloc failed, graph to big?\n");
    return(0);
  }
  for (i=0 ; i<(Mu) ; i++) invu[i]=-1;   
  for (i=0 ; i<N ; i++) invu[u[i]]=i;    
 
  /* Search for neighbours*/
  j=0;
  for (i=0 ; i<N ; i++){ 
    /* the base polong istelf */
    A[j] = i;
    B[j] = i;
    D[j] = 0;
    j++;   
    ui = u[i];
    /*6 neighbours at  distance 1*/
    if (ui+1 < Mu)
      if(invu[ui+1] > -1){
	A[j] = i;
	B[j] = invu[ui+1];
	D[j] = 1;
	j++;
      }
    if (ui > 0)
      if(invu[ui-1] > -1){
	A[j] = i;
	B[j] = invu[ui-1];
	D[j] = 1;
	j++;
      }
    if (ui+Mx < Mu)
      if(invu[ui+Mx]>-1){
	A[j] = i;
	B[j] = invu[ui+Mx];
	D[j] = 1;
	j++;
      }
    if (ui+1 > Mx)
      if(invu[ui-Mx]>-1){
	A[j] = i;
	B[j] = invu[ui-Mx];
	D[j] = 1;
	j++;
      }
    if (ui+Mxy <Mu )
     if(invu[ui + Mxy]>-1){
      A[j] = i;
      B[j] = invu[ui + Mxy];
      D[j] = 1;
      j++;
    }
    if (ui+1>Mxy)
      if(invu[ui-Mxy]>-1){
	A[j] = i;
	B[j] = invu[ui-Mxy];
	D[j] = 1;
	j++;
      }
      /*12 neighbours at sqrt(2) distance*/
    if (ui+Mx+1 < Mu)
      if(invu[ui+1+Mx]>-1){
	A[j] = i;
	B[j] = invu[ui+1+Mx];
	D[j] = _SQRT2;
	j++;
      }
    if (ui>Mx)
      if(invu[ui-1-Mx]>-1){
	A[j] = i;
	B[j] = invu[ui-1-Mx];
	D[j] = _SQRT2;
	j++;
      }
    if (ui+Mx<Mu+1)
      if(invu[ui-1+Mx]>-1){
	A[j] = i;
	B[j] = invu[ui-1+Mx];
	D[j] = _SQRT2;
	j++;
      }
    if (ui+2>Mx)
      if(invu[ui+1-Mx]>-1){
	A[j] = i;
	B[j] = invu[ui+1-Mx];
	D[j] = _SQRT2;
	j++;
      }
    if (ui+Mxy+1<Mu)
      if(invu[ui+1+Mxy]>-1){
	A[j] = i;
	B[j] = invu[ui+1+Mxy];
	D[j] = _SQRT2;
	j++;
      }
    if (ui>Mxy)
      if(invu[ui-1-Mxy]>-1){
	A[j] = i;
	B[j] = invu[ui-1-Mxy];
	D[j] = _SQRT2;
	j++;
      }
    if (ui+Mxy<Mu+1)
      if(invu[ui-1+Mxy]>-1){
	A[j] = i;
	B[j] = invu[ui-1+Mxy];
	D[j] = _SQRT2;
	j++;
      }
    if (ui+2>Mxy)
      if(invu[ui+1-Mxy]>-1){
	A[j] = i;
	B[j] = invu[ui+1-Mxy];
	D[j] = _SQRT2;
	j++;
      }
    if (ui+Mx+Mxy<Mu)
      if(invu[ui+Mxy+Mx]>-1){
	A[j] = i;
	B[j] = invu[ui+Mxy+Mx];
	D[j] = _SQRT2;
	j++;
      }
    if (ui+1>Mx+Mxy)
      if(invu[ui-Mxy-Mx]>-1){
	A[j] = i;
	B[j] = invu[ui-Mxy-Mx];
	D[j] = _SQRT2;
	j++;
      }
    if (ui+Mxy<Mu+Mx)
      if(invu[ui+Mxy-Mx]>-1){
	A[j] = i;
	B[j] = invu[ui+Mxy-Mx];
	D[j] = _SQRT2;
	j++;
      }
    if (ui+Mx+1>+Mxy)
      if(invu[ui-Mxy+Mx]>-1){
	A[j] = i;
	B[j] = invu[ui-Mxy+Mx];
	D[j] = _SQRT2;
	j++;
      }
  }

  E = j;
  thisone = fff_graph_build(N,E,A,B,D);
  
  *G = thisone;

  free(u);
  free(invu);
  free(A);
  free(B);
  free(D);
  return(E);
} 


long fff_graph_grid_twenty_six(fff_graph** G, const long* xyz, const long N)
{
  /*  char* proc = "fff_graph_grid_twenty_six"; */
  
  fff_graph * thisone;
  
  long E = 0;
  long i,j;
  long Mx,Mxy,Mu,ui;
  long *u, *A, *B, *invu; 
  double *D; 

  u = ( long*) calloc( N,sizeof(long));
  if (!u) return(0);
  A = ( long*) calloc( N*27,sizeof(long));
  if (!A) return(0);
  B = ( long*) calloc( N*27,sizeof(long));
  if (!B) return(0);
  D = (double*) calloc( N*27,sizeof(double));
  if (!D) return(0);

  _fff_graph_preprocess_grid(u,&Mx,&Mxy, &Mu,N,xyz);
 
  /* find  invu such that  invu[u[i]]=i */
  invu = (long*) calloc( Mu,sizeof(long));
  if (!invu) return(0);
  for (i=0 ; i<(Mu) ; i++) invu[i]=-1;   
  for (i=0 ; i<N ; i++) invu[u[i]]=i;    
 
  /* Search for neighbours*/
  j=0;
  for (i=0 ; i<N ; i++){ 
    /* the base polong istelf */
    A[j] = i;
    B[j] = i;
    D[j] = 0;
    j++;   
    ui = u[i];
    /*6 neighbours at  distance 1*/
    if (ui+1 < Mu)
      if(invu[ui+1] > -1){
	A[j] = i;
	B[j] = invu[ui+1];
	D[j] = 1;
	j++;
      }
    if (ui > 0)
      if(invu[ui-1] > -1){
	A[j] = i;
	B[j] = invu[ui-1];
	D[j] = 1;
	j++;
      }
    if (ui+Mx < Mu)
      if(invu[ui+Mx]>-1){
	A[j] = i;
	B[j] = invu[ui+Mx];
	D[j] = 1;
	j++;
      }
    if (ui+1 > Mx)
      if(invu[ui-Mx]>-1){
	A[j] = i;
	B[j] = invu[ui-Mx];
	D[j] = 1;
	j++;
      }
    if (ui+Mxy<Mu)
     if(invu[ui + Mxy]>-1){
      A[j] = i;
      B[j] = invu[ui + Mxy];
      D[j] = 1;
	j++;
    }
    if (ui+1>Mxy)
      if(invu[ui-Mxy]>-1){
	A[j] = i;
	B[j] = invu[ui-Mxy];
	D[j] = 1;
	j++;
      }
      /*12 neighbours at sqrt(2) distance*/
    
    if (ui+Mx+1<Mu)
      if(invu[ui+1+Mx]>-1){
	A[j] = i;
	B[j] = invu[ui+1+Mx];
	D[j] = _SQRT2;
	j++;
      }
    if (ui>Mx)
      if(invu[ui-1-Mx]>-1){
	A[j] = i;
	B[j] = invu[ui-1-Mx];
	D[j] = _SQRT2;
	j++;
      }
    if (ui+Mx<Mu+1)
      if(invu[ui-1+Mx]>-1){
	A[j] = i;
	B[j] = invu[ui-1+Mx];
	D[j] = _SQRT2;
	j++;
      }
    if (ui+2>Mx)
      if(invu[ui+1-Mx]>-1){
	A[j] = i;
	B[j] = invu[ui+1-Mx];
	D[j] = _SQRT2;
	j++;
      }
    if (ui+Mxy+1<Mu)
      if(invu[ui+1+Mxy]>-1){
	A[j] = i;
	B[j] = invu[ui+1+Mxy];
	D[j] = _SQRT2;
	j++;
      }
    if (ui>Mxy)
      if(invu[ui-1-Mxy]>-1){
	A[j] = i;
	B[j] = invu[ui-1-Mxy];
	D[j] = _SQRT2;
	j++;
      }
    if (ui+Mxy<Mu+1)
      if(invu[ui-1+Mxy]>-1){
	A[j] = i;
	B[j] = invu[ui-1+Mxy];
	D[j] = _SQRT2;
	j++;
      }
    if (ui+2>Mxy)
      if(invu[ui+1-Mxy]>-1){
	A[j] = i;
	B[j] = invu[ui+1-Mxy];
	D[j] = _SQRT2;
	j++;
	}
    if (ui+Mx+Mxy<Mu)
      if(invu[ui+Mxy+Mx]>-1){
	A[j] = i;
	B[j] = invu[ui+Mxy+Mx];
	D[j] = _SQRT2;
	j++;
	}
    if (ui+1>Mx+Mxy)
      if(invu[ui-Mxy-Mx]>-1){
	A[j] = i;
	B[j] = invu[ui-Mxy-Mx];
	D[j] = _SQRT2;
	j++;
      }
    if (ui+Mxy<Mu+Mx)
      if(invu[ui+Mxy-Mx]>-1){
	A[j] = i;
	B[j] = invu[ui+Mxy-Mx];
	D[j] = _SQRT2;
	j++;
      }
    if (ui+Mx+1>+Mxy)
      if(invu[ui-Mxy+Mx]>-1){
	A[j] = i;
	B[j] = invu[ui-Mxy+Mx];
	D[j] = _SQRT2;
	j++;
	}
       
    /*8 neighbours at srt(3) distance*/
    if (ui+Mxy+1<Mu+Mx)
      if(invu[ui+Mxy-Mx+1]>-1){
	A[j] = i;
	B[j] = invu[ui+Mxy-Mx+1];
	D[j] = _SQRT3;
	j++;
      }
    if (ui+Mxy<Mu+Mx+1)
      if(invu[ui+Mxy-Mx-1]>-1){
	A[j] = i;
	B[j] = invu[ui+Mxy-Mx-1];
	D[j] = _SQRT3;
	j++;
      }
    if (ui+Mxy+Mx<Mu+1)
      if(invu[ui+Mxy+Mx-1]>-1){
	A[j] = i;
	B[j] = invu[ui+Mxy+Mx-1];
	D[j] = _SQRT3;
	j++;
      }
    if (ui+Mxy+Mx+1<Mu)
      if(invu[ui+Mxy+Mx+1]>-1){
	A[j] = i;
	B[j] = invu[ui+Mxy+Mx+1];
	D[j] = _SQRT3;
	j++;
      } 
      
    if (ui+Mx>Mxy)
      if(invu[ui-Mxy+Mx-1]>-1){
	A[j] = i;
	B[j] = invu[ui-Mxy+Mx-1];
	D[j] = _SQRT3;
	j++;
      }
    if (ui>Mxy+Mx)
      if(invu[ui-Mxy-Mx-1]>-1){
	A[j] = i;
	B[j] = invu[ui-Mxy-Mx-1];
	D[j] = _SQRT3;
	j++;
      }
    if (ui+Mx+2>Mxy)
      if(invu[ui-Mxy+Mx+1]>-1){
	A[j] = i;
	B[j] = invu[ui-Mxy+Mx+1];
	D[j] = _SQRT3;
	j++;
      }
    if (ui+2>Mxy+Mx)
      if(invu[ui-Mxy-Mx+1]>-1){
	A[j] = i;
	B[j] = invu[ui-Mxy-Mx+1];
	D[j] = _SQRT3;
	j++;
      } 
    
  }
  E = j;
  thisone = fff_graph_build(N,E,A,B,D);
  
  *G = thisone;

  free(u);
  free(invu);
  free(A);
  free(B);
  free(D);
  return(E);
} 

static void _fff_graph_preprocess_vgrid( long*u, long*MMx, long* MMxy, long* MMu,  const fff_array* xyz)
{
  /*   char* proc = "fff_graph_preprocess_grid"; */
  long i;
  long mx,my,mz,Mx,My,Mxy,Mu;
  long N = xyz->dimX;

  /* Find minimal/maximal values of (x,y,z) */
  mx = fff_array_get2d(xyz,0,0);
  my = fff_array_get2d(xyz,0,1);
  mz = fff_array_get2d(xyz,0,2);
  Mx = fff_array_get2d(xyz,0,0);
  My = fff_array_get2d(xyz,0,1);
  
  for (i=0 ; i<N ; i++){
    if (fff_array_get2d(xyz,i,0)<mx) mx = fff_array_get2d(xyz,i,0);
    if (fff_array_get2d(xyz,i,1)<my) my = fff_array_get2d(xyz,i,1);
    if (fff_array_get2d(xyz,i,2)<mz) mz = fff_array_get2d(xyz,i,2);   
    if (fff_array_get2d(xyz,i,0)>Mx) Mx = fff_array_get2d(xyz,i,0);
    if (fff_array_get2d(xyz,i,1)>My) My = fff_array_get2d(xyz,i,1); 
    /* printf(" %d ",fff_array_get2d(xyz,i,1)); */
  }
  /* printf(" %d %d %d %d %d \n",mx,my,mz,Mx,My);*/
  
  Mx = Mx-mx+2;
  My = My-my+2;
  Mxy = Mx*My;
  Mu = 0;

 

  /* Code (x,y,z) by a scalar u*/
  for (i=0 ; i<N ; i++){
    u[i] = fff_array_get2d(xyz,i,0)-mx + (fff_array_get2d(xyz,i,1)-my)*Mx + (fff_array_get2d(xyz,i,2)-mz)*Mxy;
    if (u[i]>Mu) Mu = u[i];
  }
  Mu = Mu+1;  
  
  *MMx = Mx;
  *MMxy = Mxy;
  *MMu = Mu;
}

long fff_graph_grid(fff_graph** G, const fff_array* xyz, const long k)
{
  
  fff_graph * thisone;  
  long E = 0;
  long i,j;
  long Mx,Mxy,Mu,ui;
  long q = 6;
  long *u, *A, *B, *invu; 
  double *D; 

  /* printf("%d %d \n",xyz->size1,xyz->size2); */
  
  /* argument checking */
  long N = xyz->dimX;
  if (( xyz->dimY !=3)||(N<1)){
    FFF_WARNING("Incorrect grid matrix supplied\n");
    FFF_ERROR("Incorrect grid matrix supplied\n", EDOM);
    return(0);
  }
  if ((k!=6)&&(k!=18)&&(k!=26)) {
    FFF_WARNING("Wrong value for k. Corrected to k=6\n");
  }
  else
    q=k;
  
  
  u = ( long*) calloc( N,sizeof(long));
  if (!u) {
    FFF_WARNING(" calloc failed. The graph is too big?");
    return(0);
  }
  A = ( long*) calloc( N*(q+1),sizeof(long));
  if (!A) {
    FFF_WARNING(" calloc failed. The graph is too big?");
    return(0);
  }
  B = ( long*) calloc( N*(q+1),sizeof(long));
  if (!B) {
    FFF_WARNING(" calloc failed. The graph is too big?");
    return(0);
  }
  D = (double*) calloc( N*(q+1),sizeof(double));
  if (!D) {
    FFF_WARNING(" calloc failed. The graph is too big?");
    return(0);
  }

  _fff_graph_preprocess_vgrid(u,&Mx,&Mxy, &Mu,xyz);
 
  /* find  invu such that  invu[u[i]]=i */
  invu = (long*) calloc( Mu,sizeof(long));
  if (!invu) {
    FFF_WARNING(" calloc failed. The graph is too big?");
    return(0);
  }
  for (i=0 ; i<(Mu) ; i++) invu[i]=-1;   
  for (i=0 ; i<N ; i++) invu[u[i]]=i;    
 
  /* Search for neighbours*/
  j=0;
  for (i=0 ; i<N ; i++){ 
    /* the base polong istelf */
    A[j] = i;
    B[j] = i;
    D[j] = 0;
    j++;   
    ui = u[i];
    /*6 neighbours at  distance 1*/
    if (ui+1 < Mu)
      if(invu[ui+1] > -1){
	A[j] = i;
	B[j] = invu[ui+1];
	D[j] = 1;
	j++;
      }
    if (ui > 0)
      if(invu[ui-1] > -1){
	A[j] = i;
	B[j] = invu[ui-1];
	D[j] = 1;
	j++;
      }
    if (ui+Mx < Mu)
      if(invu[ui+Mx]>-1){
	A[j] = i;
	B[j] = invu[ui+Mx];
	D[j] = 1;
	j++;
      }
    if (ui+1 > Mx)
      if(invu[ui-Mx]>-1){
	A[j] = i;
	B[j] = invu[ui-Mx];
	D[j] = 1;
	j++;
      }
    if (ui+Mxy<Mu)
     if(invu[ui + Mxy]>-1){
      A[j] = i;
      B[j] = invu[ui + Mxy];
      D[j] = 1;
	j++;
    }
    if (ui+1>Mxy)
      if(invu[ui-Mxy]>-1){
	A[j] = i;
	B[j] = invu[ui-Mxy];
	D[j] = 1;
	j++;
      }
    if (q>6){
      /*12 neighbours at sqrt(2) distance*/
      
      if (ui+Mx+1<Mu)
	if(invu[ui+1+Mx]>-1){
	  A[j] = i;
	  B[j] = invu[ui+1+Mx];
	  D[j] = _SQRT2;
	  j++;
	}
      if (ui>Mx)
	if(invu[ui-1-Mx]>-1){
	  A[j] = i;
	  B[j] = invu[ui-1-Mx];
	  D[j] = _SQRT2;
	  j++;
	}
      if (ui+Mx<Mu+1)
	if(invu[ui-1+Mx]>-1){
	  A[j] = i;
	  B[j] = invu[ui-1+Mx];
	  D[j] = _SQRT2;
	  j++;
	}
      if (ui+2>Mx)
	if(invu[ui+1-Mx]>-1){
	  A[j] = i;
	  B[j] = invu[ui+1-Mx];
	  D[j] = _SQRT2;
	  j++;
	}
      if (ui+Mxy+1<Mu)
	if(invu[ui+1+Mxy]>-1){
	  A[j] = i;
	  B[j] = invu[ui+1+Mxy];
	  D[j] = _SQRT2;
	  j++;
	}
      if (ui>Mxy)
	if(invu[ui-1-Mxy]>-1){
	  A[j] = i;
	  B[j] = invu[ui-1-Mxy];
	  D[j] = _SQRT2;
	  j++;
	}
      if (ui+Mxy<Mu+1)
	if(invu[ui-1+Mxy]>-1){
	  A[j] = i;
	  B[j] = invu[ui-1+Mxy];
	  D[j] = _SQRT2;
	  j++;
	}
      if (ui+2>Mxy)
	if(invu[ui+1-Mxy]>-1){
	  A[j] = i;
	  B[j] = invu[ui+1-Mxy];
	  D[j] = _SQRT2;
	  j++;
	}
      if (ui+Mx+Mxy<Mu)
	if(invu[ui+Mxy+Mx]>-1){
	  A[j] = i;
	  B[j] = invu[ui+Mxy+Mx];
	  D[j] = _SQRT2;
	  j++;
	}
      if (ui+1>Mx+Mxy)
	if(invu[ui-Mxy-Mx]>-1){
	  A[j] = i;
	  B[j] = invu[ui-Mxy-Mx];
	  D[j] = _SQRT2;
	  j++;
	}
      if (ui+Mxy<Mu+Mx)
	if(invu[ui+Mxy-Mx]>-1){
	  A[j] = i;
	  B[j] = invu[ui+Mxy-Mx];
	  D[j] = _SQRT2;
	  j++;
	}
      if (ui+Mx+1>+Mxy)
	if(invu[ui-Mxy+Mx]>-1){
	  A[j] = i;
	  B[j] = invu[ui-Mxy+Mx];
	  D[j] = _SQRT2;
	  j++;
	}
      if (q>18){
	/*8 neighbours at srt(3) distance*/
	if (ui+Mxy+1<Mu+Mx)
	  if(invu[ui+Mxy-Mx+1]>-1){
	    A[j] = i;
	    B[j] = invu[ui+Mxy-Mx+1];
	    D[j] = _SQRT3;
	    j++;
	  }
	if (ui+Mxy<Mu+Mx+1)
	  if(invu[ui+Mxy-Mx-1]>-1){
	    A[j] = i;
	    B[j] = invu[ui+Mxy-Mx-1];
	    D[j] = _SQRT3;
	    j++;
	  }
	if (ui+Mxy+Mx<Mu+1)
	  if(invu[ui+Mxy+Mx-1]>-1){
	    A[j] = i;
	    B[j] = invu[ui+Mxy+Mx-1];
	    D[j] = _SQRT3;
	    j++;
	  }
	if (ui+Mxy+Mx+1<Mu)
	  if(invu[ui+Mxy+Mx+1]>-1){
	    A[j] = i;
	    B[j] = invu[ui+Mxy+Mx+1];
	    D[j] = _SQRT3;
	    j++;
	  } 
	
	if (ui+Mx>Mxy)
	  if(invu[ui-Mxy+Mx-1]>-1){
	    A[j] = i;
	    B[j] = invu[ui-Mxy+Mx-1];
	    D[j] = _SQRT3;
	    j++;
	  }
	if (ui>Mxy+Mx)
	  if(invu[ui-Mxy-Mx-1]>-1){
	    A[j] = i;
	    B[j] = invu[ui-Mxy-Mx-1];
	    D[j] = _SQRT3;
	    j++;
	  }
	if (ui+Mx+2>Mxy)
	  if(invu[ui-Mxy+Mx+1]>-1){
	    A[j] = i;
	    B[j] = invu[ui-Mxy+Mx+1];
	    D[j] = _SQRT3;
	    j++;
	  }
	if (ui+2>Mxy+Mx)
	  if(invu[ui-Mxy-Mx+1]>-1){
	    A[j] = i;
	    B[j] = invu[ui-Mxy-Mx+1];
	    D[j] = _SQRT3;
	    j++;
	  } 
      }
    }
  }
  E = j;
  
  thisone = fff_graph_build(N,E,A,B,D);
  if (thisone == NULL) {
    FFF_WARNING("fff_graph_build failed");
    return(-1);
  }
 
  *G = thisone;

  free(u);
  free(invu);
  free(A);
  free(B);
  free(D);
  return(E);
} 



/**********************************************************************
 ******************************* MST **********************************
**********************************************************************/

double fff_graph_MST(fff_graph* G,const fff_matrix* X)
{ 
  /* char* proc = "fff_graph_MST"; */
  long V = X->size1;
  long T = X->size2;
  double ndist,auxdist,dx,maxdist;
  long nnbcc,nbcc = V;
  double * mindist;
  long * amd;
  long * imd;
  long i,n1,n2,j,t,k;
  double length = 0;
  long *idx, *label; 
  long q; 

  /* labels Initialization */
  idx = (long*) calloc( V,sizeof(long));
  label = (long*) calloc( V,sizeof(long));
  if (!label) return(0);
  for (i =0; i<V; i++) label[i]=i;
  
  /* init maxdist */
  maxdist = 0;
  for (n1 =1; n1<V; n1++){
    ndist = 0;
    for ( t=0 ; t<T ; t++){
      dx = fff_matrix_get(X,n1,t)-fff_matrix_get(X,0,t);
      ndist += dx*dx;
    }
    if (ndist>maxdist) maxdist = ndist; 
  }   
  maxdist += 1.e-7;
  
  q = 0;
  mindist = (double*) calloc( nbcc,sizeof(double));
  if (!mindist)    return(0);
  amd = (long*) calloc( nbcc,sizeof(long));
  if (!amd)    return(0);
  imd = (long*) calloc( nbcc,sizeof(long));
  if (!imd)    return(0);
  
  while (nbcc>1){  
    for (i=0; i<nbcc; i++){
      idx[i]=i;
      mindist[i] = maxdist;
    }

    /* for each connected component, find the minimal single link */    
    
    for (n1=0 ; n1<V ; n1++){
      j = label[n1];
      for ( n2=0 ; n2<n1 ; n2++){
		k = label[n2];
		if (j!=k){
		  auxdist = FFF_MAX(mindist[j],mindist[k]);
		  ndist = 0;
		  for ( t=0 ; t<T ; t++){
			dx = fff_matrix_get(X,n1,t)-fff_matrix_get(X,n2,t);
			ndist += dx*dx;
			if (ndist>auxdist) break;
		  }
		  if (ndist<mindist[j]) {
			mindist[j] = ndist;
			amd[j] = n1;
			imd[j] = n2;
		  }
		  if (ndist<mindist[k]){
			mindist[k] = ndist;
			amd[k] = n2;
			imd[k] = n1;
		  }
		}
      }
    }
    
    /* write the new edges at the current iteration */
    nnbcc = nbcc;
    
    
    for(i=0; i<nnbcc; i++){
      k = label[amd[i]];
      while (k > idx[k])
		k  = idx[k];
      
      j = label[imd[i]];
      while (j > idx[j])
		j  = idx[j];
      if (k!=j){
		ndist = sqrt(mindist[i]);
		G->eA[q] = amd[i];
		G->eB[q] = imd[i];
		G->eD[q] = ndist;
		q++;
		G->eA[q] = imd[i];
		G->eB[q] = amd[i];
		G->eD[q] = ndist;
		q++;
		
		if (k<j)
		  idx[j] = k;
		else
		  idx[k] = j;
		
		nbcc--;
		length += ndist; 
      } 
    }
    
    /* relabel the ccs */
    nnbcc = fff_graph_cc_label(label,G);
  } 
  free(mindist);
  free(amd);
  free(imd);
  free(label);
  free(idx);
  return(length);
}

double fff_graph_skeleton(fff_graph* K, const fff_graph* G)
{
  /* For now G is assumed to be connected */
  /* char* proc = "fff_graph_MST"; */
  long V = G->V;
  double ndist,maxdist;
  long nnbcc,nbcc = V;
  double * mindist;
  long * amd;
  long * imd;
  long i,n1,n2,j,e,k;
  double length = 0;
  long *idx, *label; 
  long q; 

  /* labels Initialization */
  idx = (long*) calloc( V,sizeof(long));
  label = (long*) calloc( V,sizeof(long));
  if (!label) return(0);
  for (i =0; i<V; i++) label[i]=i;
  
  /* init maxdist */
  maxdist = 0;
  for (e=0 ; e<G->E ; e++)
	if (G->eD[e]>maxdist)
	  maxdist = G->eD[e];
  
  maxdist += 1.e-7;
  
  q = 0;
  mindist = (double*) calloc( nbcc,sizeof(double));
  if (!mindist)    return(0);
  amd = (long*) calloc( nbcc,sizeof(long));
  if (!amd)    return(0);
  imd = (long*) calloc( nbcc,sizeof(long));
  if (!imd)    return(0);
  
  while (nbcc>1){  
    for (i=0; i<nbcc; i++){
      idx[i]=i;
      mindist[i] = maxdist;
    }
    /* for each connected component, find the minimal single link */
	for (e=0 ; e<G->E ; e++){
	  n1 = G->eA[e];
	  n2 = G->eB[e];
	  j = label[n1];
	  k = label[n2];
	  if (j!=k){
		 ndist = G->eD[e];
		 if (ndist<mindist[j]) {
			mindist[j] = ndist;
			amd[j] = n1;
			imd[j] = n2;
		  }
		  if (ndist<mindist[k]){
			mindist[k] = ndist;
			amd[k] = n2;
			imd[k] = n1;
		  }
		}
      }
    
    /* write the new edges at the current iteration */
  nnbcc = nbcc;
  for(i=0; i<nnbcc; i++){
	k = label[amd[i]];
	while (k > idx[k])
	  k  = idx[k];
      
	j = label[imd[i]];
	while (j > idx[j])
	  j  = idx[j];
	if (k!=j){
	  ndist = mindist[i];
	  K->eA[q] = amd[i];
	  K->eB[q] = imd[i];
	  K->eD[q] = ndist;
	  q++;
	  K->eA[q] = imd[i];
	  K->eB[q] = amd[i];
	  K->eD[q] = ndist;
		q++;
		
		if (k<j)
		  idx[j] = k;
		else
		  idx[k] = j;
		
		nbcc--;
		length += ndist; 
      } 
    }
    
    /* relabel the ccs */
    nnbcc = fff_graph_cc_label(label,K);
 
  } 
free(mindist);
  free(amd);
  free(imd);
  free(label);
  free(idx);
  return(length);
}



/**********************************************************************
 *************************** cc analysis ******************************
**********************************************************************/

extern long fff_graph_to_neighb(fff_array *cindices, fff_array * neighb, fff_vector* weight, const fff_graph* G)
{
  long V = G->V;
  long E = G->E;
  if (((cindices->dimX)!=V+1)|((neighb->dimX)!=E)|((weight->size)!=E)){
    FFF_ERROR("inconsistant vector size \n",EDOM);
    /* return(1); */
  }
   _fff_graph_vect_neighb(cindices,neighb,weight,G);
   return(0);
}

static long _fff_graph_vect_neighb( fff_array *cindices, fff_array * neighb, fff_vector* weight, const fff_graph* G)
{
  /* This function recomputes the connectivity system of the graph in an efficient way:
  The edges are arraned in the following manner 
       origins =  [0..0 1..1 .. V-1..V-1]
       ends    =  [neignb[0].. neighb[E-1]]
       weight  =  [weights[0]..weights[E-1]]
       
       cindices codes for the origin vector: origin=i between cindices[i] and cindices[i+1]-1 
  */
  long E = G->E;
  long V = G->V;
  long a,b;
  long i,j;
  double aux = 0; 

  if (((cindices->dimX)<V)|((neighb->dimX)<E)|((weight->size)<E)){
    FFF_ERROR("inconsistant vector size \n",EDOM);
    /* return(1); */
  }
  
  fff_array_set_all( cindices,0 );

  for(i=0 ; i<E ; i++){
    j = fff_array_get1d(cindices,G->eA[i])+1;
    fff_array_set1d(cindices,G->eA[i],j);
    fff_array_set1d(neighb,i,-1);
  }
  
  for(i=0; i<V; i++){
    j = fff_array_get1d(cindices,i);
    aux  = aux + j;
    fff_array_set1d(cindices,i, aux - j);
  } 
  if ((cindices->dimX)>V)
    fff_array_set1d(cindices,V, E);

  for(i=0 ; i<E ; i++){
    a = G->eA[i]; 
    b = G->eB[i]; 
    j = fff_array_get1d(cindices,a);
    while (fff_array_get1d(neighb,j)>-1) j++;
    fff_array_set1d(neighb,j,b);
    fff_vector_set(weight,j,G->eD[i]);
  } 
  return(0);
}

long fff_graph_cc_label( long* label, const fff_graph* G)
{ 
  long n1,ne;
  long k;
  long E = G->E;
  long N = G->V;
  long remain = N;
  long su,sv;
  
  for (n1 =0; n1<N; n1++) 
    label[n1] = -1;
  
  k=0;
  while (remain>0){
    n1 = 0;
    while (label[n1]>-1) n1++;
    label[n1] = k;
    su =  0;
    sv  = 1;
	
    while (sv>su){
      su = sv; 
      for(ne=0; ne<E; ne++){
		if(label[G->eA[ne]]==k)
		  label[G->eB[ne]] = k;
		if(label[G->eB[ne]]==k)
		  label[G->eA[ne]] = k;
      }
      sv = 0;
      for(n1=0; n1<N; n1++) 
		sv += (label[n1]==k); 
    }
	
    remain = remain-su;
    k++;  
  }
  return(k);
}

/**********************************************************************
 *************************** Dijkstra, Floyd ******************************
**********************************************************************/

long fff_graph_dijkstra( double *dist, const fff_graph* G, const long seed)
{
  long i,E=G->E;
  double infdist = 1.0;
  for (i=0 ; i<E ; i++)
    if (G->eD[i]<0){
      FFF_WARNING("found a negative distance \n");
      return(1);
    }
    else
      infdist += G->eD[i];
  fff_graph_Dijkstra( dist,G,seed,infdist);
  return(0);
    
}

long fff_graph_Dijkstra( double *dist, const fff_graph* G, const long seed, const double infdist)
{ 
  /* char* proc = "fff_graph_Dijkstra"; */
  long E = G->E;
  long V = G->V;
  long i,j,k,l,win,start,end;
  double newdist;
  
  fff_vector *dg = fff_vector_new(V);
  fff_array *lg = fff_array_new1d(FFF_LONG,V);
  fff_array *cindices = fff_array_new1d(FFF_LONG,V+1);
  fff_array *neighb = fff_array_new1d(FFF_LONG,E);
  fff_vector *weight = fff_vector_new(E);
  long *lgdata = (long*) lg->data;
  long *cidata = (long*) cindices->data;
  long *nedata = (long*) neighb->data;
  long ri = _fff_graph_vect_neighb(cindices,neighb,weight,G);
  

  /* initializations*/
  for(i=0 ; i<V ; i++){
    dist[i] = infdist;
    dg->data[i] = infdist;
    lgdata[i] = -1;
  }
  win = seed;
  dist[win] = 0;
  dg->data[0] = 0;
  lgdata[0] = win;
  k = 1;
  
  /* iterations */
  for (j=1 ; j<V ; j++){
    start = cidata[win];
    end = cidata[win+1];
    for (i=start ; i<end ; i++){
      l = nedata[i];
      if (dist[win]+weight->data[i] < dist[l]){
	  newdist = dist[win] + weight->data[i];
	  if (dist[l] < infdist)
	    ri += _fff_list_move(lgdata, dg->data, l, newdist, k);
	  else{
	    ri += _fff_list_add(lgdata, dg->data, l, newdist, k);
	    k++;
	  }
	  dist[l] = newdist;
      }
    } 
    win = lgdata[j];
    if (win == -1) break;
  }

  fff_array_delete(cindices);
  fff_array_delete(neighb);
  fff_vector_delete(dg);
  fff_array_delete(lg);
  fff_vector_delete(weight);
  return(ri);
}

int fff_graph_Dijkstra_multiseed( fff_vector *dist, const fff_graph* G, const fff_array* seeds)
{ 
  long E = G->E;
  long V = G->V;
  long i,j,k,l,win,start,end;
  double newdist;
  int sp = seeds->dimX;
  double infdist = FFF_POSINF;
  
  fff_vector *dg = fff_vector_new(V);
  fff_array *lg = fff_array_new1d(FFF_LONG,V);
  fff_array *cindices = fff_array_new1d(FFF_LONG,V+1);
  fff_array *neighb = fff_array_new1d(FFF_LONG,E);
  fff_vector *weight = fff_vector_new(E);
  long *cidata = (long*) cindices->data;
  long *lgdata = (long*) lg->data;
  long *nedata = (long*) neighb->data;
  double dsmin,dsmax;
  long smin, smax, ri; 

  for (i=0 ; i<E ; i++)
    if (G->eD[i]<0){
      FFF_WARNING("found a negative distance \n");
      return(1);
    }

  fff_array_extrema ( &dsmin, &dsmax, seeds );
  smin = (long) dsmin;
  smax = (long) dsmax;
 
  if ((smin<0)|(smax>V-1)){
    FFF_ERROR("seeds have incorrect indices \n",EDOM);
    return(1);
  }
  ri = _fff_graph_vect_neighb(cindices,neighb,weight,G);

  /* initializations*/

  for(i=0 ; i<V ; i++){
    fff_vector_set(dg,i,infdist);
    fff_array_set1d(lg,i,-1);
	fff_vector_set(dist,i,infdist);
   }
  k = 0;
  for(i=0 ; i<sp ; i++){
    win = fff_array_get1d(seeds,i);
    if (fff_vector_get(dist,win)>0) k++;
    fff_vector_set(dist,win,0);
    fff_vector_set(dg,i,0);
    fff_array_set1d(lg,i,win);
   } 
  win = fff_array_get1d(lg,0);
    
  /* iterations */
  for (j=1 ; j<V ; j++){
    start = cidata[win];
    end = cidata[win+1];
    for (i=start ; i<end ; i++){
      l = nedata[i];
      if (fff_vector_get(dist,win)+fff_vector_get(weight,i) < fff_vector_get(dist,l)){
		newdist = fff_vector_get(dist,win) + fff_vector_get(weight,i);
		if (fff_vector_get(dist,l) < infdist)
		  ri += _fff_list_move(lgdata, dg->data, l, newdist, k);
		else{
		  ri += _fff_list_add(lgdata, dg->data, l, newdist, k);
		  k++;
		}
		fff_vector_set(dist,l,newdist);
      }
    } 
    win = fff_array_get1d(lg,j);
    if (win == -1) break;
  }

  fff_array_delete(cindices);
  fff_array_delete(neighb);
  fff_vector_delete(dg);
  fff_array_delete(lg);
  fff_vector_delete(weight);
  return(ri);
}

long fff_graph_partial_Floyd( fff_matrix *dist, const fff_graph* G, const long *seeds)
{
  long i,j;
  long sp = dist->size1;
  double infdist = 1.0;
  long V = G->V;
  long E = G->E;
  long ri = 0;
  double *bufd; 
 
  if ((dist->size2)!=V){
	FFF_ERROR("incompatible matrix size \n",EDOM);
	/* return(1); */
    }
  
  for (i=0 ; i<E ; i++)
    if (G->eD[i]<0){
      FFF_WARNING("found a negative distance \n");
      return(1);
    }
    else
      infdist += G->eD[i];
  
  bufd = (double*) calloc(V,sizeof(double));
  
  for (i=0 ; i<sp ; i++){   
    ri = fff_graph_Dijkstra( bufd,G,seeds[i],infdist);
    for(j=0 ; j<V ; j++)
      fff_matrix_set(dist,i,j,bufd[j]);
  }
  
  free(bufd);
  return(ri);
}


long fff_graph_Floyd(fff_matrix *dist, const fff_graph* G)
{
   long i,j;
   double infdist = 1.0;
   long V = G->V;
   long E = G->E;
   long ri = 0;
   double *bufd; 

   if (((dist->size1)!=V)|((dist->size2)!=V)){
      FFF_ERROR("incompatible matrix size \n",EDOM);
      return(1);
    }
   
  for (i=0 ; i<E ; i++)
    if (G->eD[i]<0){
      FFF_WARNING("found a negative distance \n");
      return(1);
    }
    else
    infdist += G->eD[i];
  
  bufd = (double*) calloc(V,sizeof(double));

  for (i=0 ; i<V ; i++){   
    ri = fff_graph_Dijkstra( bufd,G,i,infdist);
    for(j=0 ; j<V ; j++)
      fff_matrix_set(dist,i,j,bufd[j]);
  }
  free(bufd);
  return(ri);
}


extern long fff_graph_voronoi(fff_array *label, const fff_graph* G,const  fff_array *seeds)
{
  long i,j,k,l,win,start, end;
  long sp = seeds->dimX;
  double infdist = 1.0;
  long V = G->V;
  long E = G->E;
  long ri = 0;
  double dwin,w,newdist;
  double dsmin,dsmax;
  long smin, smax; 

  fff_vector *dist, *dg, *weight; 
  fff_array *lg, *cindices, *neighb;
 
  /* argument checking */
  if ((label->dimX)!=V){
    FFF_ERROR("incompatible matrix size \n",EDOM);
    /* return(1); */
  }
  
  for (i=0 ; i<E ; i++)
    if (G->eD[i]<0){
      FFF_WARNING("found a negative distance \n");
      return(1);
    }
    else
      infdist += G->eD[i];
  
    fff_array_extrema ( &dsmin, &dsmax, seeds );
    smin = (long) dsmin;
    smax = (long) dsmax;
 
  if ((smin<0)|(smax>V-1)){
    FFF_ERROR("seeds have incorrect indices \n",EDOM);
    /* return(1); */
  }

  /* initializations*/
  
  dist = fff_vector_new(V);
  dg = fff_vector_new(V+1);
  lg = fff_array_new1d(FFF_LONG,V+1);
  cindices = fff_array_new1d(FFF_LONG,V+1);
  neighb = fff_array_new1d(FFF_LONG,E);
  weight = fff_vector_new(E);
  
  ri = _fff_graph_vect_neighb(cindices,neighb,weight,G);
  
  for(i=0 ; i<V+1 ; i++){
    fff_vector_set(dg,i,infdist);
    fff_array_set1d(lg,i,-1);
  }
  for(i=0 ; i<V ; i++){
    fff_vector_set(dist,i,infdist);
    fff_array_set1d(label,i,-1);
  }
  k = 0;
  for(i=0 ; i<sp ; i++){
    win = fff_array_get1d(seeds,i);
    if (fff_vector_get(dist,win)>0){ 
	  fff_array_set1d(lg,k,win);
	  fff_array_set1d(label,win,k);
	  k++;
	}
    fff_vector_set(dist,win,0);
    fff_vector_set(dg,i,0);
    /*fff_array_set1d(lg,i,win);*/
	/*fff_array_set1d(label,win,i);*/
  } 
  win = fff_array_get1d(seeds,0);
  /*printf("%ld %ld\n",k,win);*/

  /* iterations */
  
  for (j=1 ; j<V ; j++){	
    dwin = fff_vector_get(dist,win);
	/* printf("%d %ld %f \n",j,win,dwin); */
    start = fff_array_get1d(cindices,win);
    end = fff_array_get1d(cindices,win+1);
    
    for (i=start ; i<end ; i++){
      l = fff_array_get1d(neighb,i);
      w = fff_vector_get(weight,i);
      
      if ( dwin+w < fff_vector_get(dist,l)){
		newdist = dwin + w;	  
		if (fff_vector_get(dist,l) < infdist)
		  ri += _fff_list_move(lg->data, dg->data, l, newdist, k);
		else{
		  ri += _fff_list_add(lg->data, dg->data, l, newdist, k);
		  k++; 
		}
		fff_vector_set(dist,l,newdist);
		fff_array_set1d(label,l, fff_array_get1d(label,win));
      }
    }
    win = fff_array_get1d(lg,j);
    if (win == -1) break;
    
  }
  
  fff_array_delete(cindices);
  fff_array_delete(neighb);
  fff_vector_delete(dg);
  fff_vector_delete(dist);
  fff_array_delete(lg);
  fff_vector_delete(weight);

  return(ri);
}


/**********************************************************************
 ***** Replicator dynamics, clique extraction ************************
**********************************************************************/


extern long fff_graph_cliques(fff_array *cliques, const fff_graph* G)
{

  int i,q,V = G->V;
  double su,duw,eps = 1.e-12;
  int bstochastic = 0;
  int qmax = 1000;
  double Vl;
  int temp,k =0;
  fff_vector *u = fff_vector_new(V);
  fff_vector *v = fff_vector_new(V);
  fff_vector *w = fff_vector_new(V);
  fff_array *idx = fff_array_new1d(FFF_LONG,V);
  fff_array* invidx; 

  if ((cliques->dimX)!=V){
	FFF_ERROR("incompatible vector/graph size \n",EDOM);
  }
  
  fff_array_set_all(cliques,-1);
  
  while(fff_array_min1d(cliques)<0){
	if (bstochastic==0)
	  /* the converse case should be implemented */
	  fff_vector_set_all(u,1);
	for (i=0 ; i<V ; i++)
	  if (fff_array_get1d(cliques,i)>-1)
		fff_vector_set(u,i,0.);
	fff_vector_set_all(w,0);
	q = 0;
	duw = 1.;
	while(duw>eps){
	  fff_vector_memcpy (w,u);
	  fff_field_diffusion(u,G);
	  fff_vector_mul(u,w);
	  su = fff_vector_sum(u);
	  if (su==0) break;
	  else fff_vector_scale(u,1./su);
	  q = q+1;
	  if (q>qmax) break;
	  fff_vector_sub (w,u);
	  fff_vector_mul(w,w);
	  duw = fff_vector_sum(w);
	}
	Vl = 0;
	for (i=0 ; i<V ; i++)
	  Vl += (fff_array_get1d(cliques,i)==-1);
	if (Vl==1) Vl = 2.;
	Vl = 1./Vl;
	q = 0;
	for (i=0 ; i<V ; i++)
	  if (fff_vector_get(u,i)>Vl){
		fff_array_set1d(cliques,i,k);
		q++;
	  }
	fff_vector_set(v,k,-q);
	if (q==0) break;
	else k++;
	if ( fff_vector_sum(u))
	  break;
  }
  
  /* relabel in order to have v increasing */
  _fff_sort_vector_index(v, idx->data);
  invidx = fff_array_new1d(FFF_LONG,V);
  
  for (i=0 ; i<V ; i++)
	fff_array_set1d(invidx,fff_array_get1d(idx,i),i);
  for (i=0 ; i<V ; i++)
	if (fff_array_get1d(cliques,i)>-1){
	  temp = fff_array_get1d(invidx,fff_array_get1d(cliques,i));
	  fff_array_set1d(cliques,i,temp);
	}
	else{
	  fff_array_set1d(cliques,i,k);
	  k++;
	}

  fff_array_delete(invidx);
  fff_vector_delete(u);
  fff_vector_delete(v);
  fff_vector_delete(w);
  fff_array_delete(idx);

  return(0);
}
