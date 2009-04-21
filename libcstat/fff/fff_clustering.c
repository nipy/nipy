#include "fff_clustering.h"
#include "fff_blas.h"
#include <randomkit.h>
#include "fff_routines.h"

#include <stdlib.h>
#include <math.h>
#include <stdio.h>



static void _fff_CM_init( fff_matrix* Centers, const fff_matrix* X);
static double _fff_CM_functional(const fff_matrix* X, const fff_matrix* Centers, const fff_array *Label);
static void _fff_Mstep ( fff_array *Label, const fff_matrix* X, const fff_matrix* Centers);


static void _fff_FCM_init(fff_matrix* U);
static double _fff_FCM_functional(const fff_matrix* X, const fff_matrix* Centers, const fff_matrix *U);
static void _fff_fuzzy_Mstep ( fff_matrix *U, const fff_matrix* X, const fff_matrix* Centers);
static void _fff_fuzzy_Estep(fff_matrix* Centers, const fff_matrix* X, const fff_matrix *U);


/**********************************************************************
********************* Ward's clustering ******************************
**********************************************************************/
static int _fff_matrix_sym_get_min(long* i, long* j, double *v, const fff_matrix *M);
double _inertia(const int i,const int j, const fff_matrix* M1, const fff_matrix *M2, const long *count);

static int _fff_matrix_sym_get_min(long* i, long* j, double* v, const fff_matrix *M)
{
  /* return the coordinates and value of the matrix min */
  /* optimized through symmetry */
  long k,l,n =M->size1;
  v[0] = fff_matrix_get(M,0,0);
  i[0] = 0;
  j[0] = 0;
  for (k=0; k<n ; k++)
	for (l=0 ; l<k ; l++)
	  if (fff_matrix_get(M,k,l)<v[0]){
		v[0] = fff_matrix_get(M,k,l);
		i[0] = k;
		j[0]= l;
	  }
  return 0;
}

double _inertia(const int i,const int j, const fff_matrix* M1, const fff_matrix *M2, const long *count)
{
  long card,k,p=M1->size2;
  double u,w,v = 0;
  
  card = count[i] +count[j]; 
  for (k=0 ; k<p ; k++){
	w = fff_matrix_get(M1,i,k) + fff_matrix_get(M1,j,k);
	w /= card;
	u = fff_matrix_get(M2,i,k) + fff_matrix_get(M2,j,k);
	u /= card;
	v += (u-w*w);
  }
  return v;
}


int fff_clustering_ward(fff_array* parent,fff_vector *cost, const fff_matrix* X)
{ 
  long i,j,k,l,k0,l0,n = X->size1, p=X->size2;
  fff_matrix * Inertia = fff_matrix_new(n,n); 
  double lx, var;
  long q,lc;
  fff_matrix * M1 = fff_matrix_new(n,p);
  fff_matrix * M2 = fff_matrix_new(n,p);
  double linf;
  long * count = (long*) calloc(n, sizeof(long));

  
  /* M1 and M2 represent the cluster-wise sum and sum of square values*/
  for (i=0 ; i<n ; i++){
	for (j=0 ; j<p; j++){
	  lx = fff_matrix_get(X,i,j);
	  fff_matrix_set(M1,i,j,lx);
	  fff_matrix_set(M2,i,j,lx*lx);
	}
  }
  linf = fff_matrix_sum(M2)+1.0;
  
  /* init count*/
  for (i=0 ; i<n ; i++) count[i]=1;
  
  /* compute the inertia matrix*/
  fff_matrix_set_all(Inertia,linf);
  for (i=0 ; i<n ; i++){
	for (j=0 ; j<i; j++){
	  var = _inertia(i,j,M1, M2, count);
	  fff_matrix_set(Inertia,i,j,var);
	  fff_matrix_set(Inertia,j,i,var);
	}
  }
  
  /* init parent */
  q = 2*n-1;
  for (i=0 ; i<q; i++) fff_array_set1d(parent,i,i);
  
  /* recursive merge loop */
  for (i=0; i<n-1 ; i++){
	q = i+n;
	
	/* detect the merge */
	_fff_matrix_sym_get_min(&k, &l, &var, Inertia);
	/* since parents are quatters, the actual identity of k and l has to be found */
	k0 = k;
	l0 = l;
	while (fff_array_get1d(parent,k)!=k)
	  k = fff_array_get1d(parent,k);
	while (fff_array_get1d(parent,l)!=l)
	  l = fff_array_get1d(parent,l);

	/* perform the merge */
	/* count,clist,parent,cost */
	fff_vector_set(cost,q,var);
	fff_array_set1d(parent,k,q);
	fff_array_set1d(parent,l,q);
	
	/* update the counts */
	lc = count[k0]+count[l0];
	count[k0] = lc;

	/* update the moments */
	/* to save space, q squattes  the place of k */
	/* the place of  l is abandoned */
	for (j =0 ; j<p; j++){
	  lx = fff_matrix_get(M1,k0,j) + fff_matrix_get(M1,l0,j); 
	  fff_matrix_set(M1,k0,j,lx);
	  var = fff_matrix_get(M2,k0,j) + fff_matrix_get(M2,l0,j);
	  fff_matrix_set(M2,k0,j,var);
	}
	
	/* update the inertia */
	for (j=0 ; j<n ; j++){
	  fff_matrix_set(Inertia,l0,j,linf);
	  fff_matrix_set(Inertia,j,l0,linf);
	}
	for (j=0 ; j<n ; j++)
	  if (fff_matrix_get(Inertia,k0,j)<linf){
		var = _inertia(k0,j,M1, M2, count);
		fff_matrix_set(Inertia,k0,j,var);
		fff_matrix_set(Inertia,j,k0,var);
	  }
  }
  
  /* delete: count,Inertia,M1,M2 */
  fff_matrix_delete(M1);
  fff_matrix_delete(M2);
  fff_matrix_delete(Inertia);
  free(count);
  return 0;
}


/**********************************************************************
********************* C-Means clustering ******************************
**********************************************************************/


double fff_clustering_cmeans( fff_matrix* Centers, fff_array *Label, const fff_matrix* X, const int maxiter,  double delta) 
{
  double J = -1;
  
  char* proc = "fff_clustering_cmeans";
  int fd = X->size2;      
  long k = Centers->size1;
  int i,j,l;
  double normdC,normC; 
  double dx;
  int verbose = 0;
  
  fff_matrix* Centers_old = fff_matrix_new(Centers->size1, Centers->size2);
  
  fff_matrix_set_all( Centers_old,0);
  
  if (fff_clustering_OntoLabel(Label,k))
    fff_Estep(Centers,Label,X); 
  else
    _fff_CM_init(Centers,X);
  
  for (l=0; l<maxiter ; l++){
    /* basic algorithm  */
    _fff_Mstep(Label,X,Centers);
    fff_Estep(Centers,Label,X); 
    
    J = _fff_CM_functional(X, Centers, Label);
    if (verbose)
      printf ("%s Iter %d functional J = %f \n",proc,l,J);
    
    /* Control of convergence */
    normdC = 0;
    normC = 0;
    for (i=0 ; i<k ; i++)  
      for (j=0 ; j<fd ; j++){
	dx = fff_matrix_get(Centers_old,i,j)-fff_matrix_get(Centers,i,j);
	normdC += dx*dx;
	dx = fff_matrix_get(Centers_old,i,j);
	normC += dx*dx;
      }
    normdC = sqrt(normdC);
    normC = sqrt(normC);
    
    fff_matrix_memcpy (Centers_old, Centers);
    if (normdC<delta*normC) break;
  }  
  
  fff_matrix_delete(Centers_old);
  
  return(J);
}


/* Initialization of the Centers matrix by picking randomly points from X */ 
static void _fff_CM_init( fff_matrix* Centers, const fff_matrix* X)
{
  int k = Centers->size1;
  int N = X->size1;
  int T = X->size2;
  int i,t;
  size_t *list = calloc(k, sizeof(size_t)); 
  double temp;
  
  /* Draw k different values in the range [0..N-1] */  
  fff_rng_draw_noreplace (list, k, N); 
  
  for (i=0 ; i<k ; i++)
    for ( t=0 ; t<T ; t++){
      temp = fff_matrix_get(X,list[i],t);
       fff_matrix_set(Centers,i,t,temp); 
    } 
  
  free(list);
  return; 
}

/* functional of the CM algorithm */
static double _fff_CM_functional(const fff_matrix* X, const fff_matrix* Centers, const fff_array *Label)
{
  int n,c;
  double J = 0;
  int N = X->size1;
  int T = X->size2;
  
  fff_vector *v1 = fff_vector_new(T);
  fff_vector *v2 = fff_vector_new(T);
  
  for (n=0 ; n<N ; n++){
    c = fff_array_get1d(Label,n);
  
    fff_matrix_get_row (v1,X,n);
    fff_matrix_get_row (v2,Centers,c);
    fff_vector_sub (v2, v1);
	fff_vector_mul(v2,v2);
    J += fff_vector_sum(v2);
  }
  fff_vector_delete(v1);
  fff_vector_delete(v2);
  J /=N;
  return(J);
}


/* Mstep of the CM algo: compute hard memberships */
static void _fff_Mstep ( fff_array *Label, const fff_matrix* X, const fff_matrix* Centers)
{
   double dist,mindist;
   double index;
   int n1,n2;
   int N = X->size1;
   int T  = X->size2;
   int C = Centers->size1;
  
   fff_vector *v1 = fff_vector_new(T);
   fff_vector *v2 = fff_vector_new(T);
   fff_array_set_all(Label,0);

   for (n1=0 ; n1<N ; n1++){
     fff_matrix_get_row (v1,X,n1);
     fff_matrix_get_row (v2,Centers,0);
     fff_vector_sub (v2, v1);
	 fff_vector_mul(v2,v2);
     mindist = fff_vector_sum(v2);
 
     index = 0;
     for (n2=1 ; n2<C ; n2++){
       fff_matrix_get_row (v2,Centers,n2);
       fff_vector_sub (v2, v1);
	   fff_vector_mul(v2,v2);
       dist = fff_vector_sum(v2);
       if (dist<mindist){
	 mindist = dist;
	 index = n2;
       }
     } 
     fff_array_set1d(Label,n1,index);
   }
   fff_vector_delete(v1);
   fff_vector_delete(v2);
}

/* E step of the CM algo: update the cluster centers */
extern void fff_Estep( fff_matrix* Centers, const fff_array *Label, const fff_matrix* X)
{
   int N = X->size1;
   int C = Centers->size1;
   int n,c;

   fff_vector *v1 = fff_vector_new(X->size2);
   fff_vector *v2 = fff_vector_new(X->size2);
   fff_array *count = fff_array_new1d(FFF_LONG,C);
   fff_array_set_all(count,0);
   fff_matrix_set_all(Centers,0);
   
   for (n=0 ; n<N ; n++){
     c = fff_array_get1d(Label,n);
     fff_array_set1d(count,c,fff_array_get1d(count,c)+1);
     fff_matrix_get_row (v1,X,n);
     fff_matrix_get_row (v2,Centers,c);
     fff_vector_add (v2,v1);
     fff_matrix_set_row (Centers,c,v2);
     
   }
   
   for (c=0 ; c<C ; c++)
     if (fff_array_get1d(count,c) > 0){
       fff_matrix_get_row (v2,Centers,c);
       fff_vector_scale (v2, 1./fff_array_get1d(count,c));
       fff_matrix_set_row (Centers,c,v2);
     }
   
   fff_array_delete(count);
   fff_vector_delete(v1);
   fff_vector_delete(v2);
}


/* Mstep of the CM algo: compute hard memberships ; quicker version */
extern int fff_clustering_Voronoi ( fff_array *Label, const fff_matrix* Centers, const fff_matrix* X)
{
   double dx,dist,mindist;
   double index;
   int n1,n2,t;
   int N = X->size1;
   int T  = X->size2;
   int C = Centers->size1;
  
   fff_array_set_all(Label,0);
  
   for (n1=0 ; n1<N ; n1++){
     mindist = 0;
     for ( t=0 ; t<T ; t++){
       dx = fff_matrix_get(X,n1,t)-fff_matrix_get(Centers,0,t);
       mindist += dx*dx;
     }
     index = 0;
     for (n2=1; n2<C; n2++){
       dist = 0;
       for ( t=0 ; t<T ; t++){
	 dx = fff_matrix_get(X,n1,t)-fff_matrix_get(Centers,n2,t);
	 dist += dx*dx;
	 if (dist>mindist) break;
       }
       if (dist<mindist){
	 mindist = dist;
	 index = (double)n2;
       }
     } 
     fff_array_set1d(Label, n1, index);
  }
   return(0);
}

/**********************************************************************
********************* FCM clustering ******************************
**********************************************************************/


extern int fff_clustering_fcm( fff_matrix* Centers, fff_array *Label, const fff_matrix* X, const int maxiter, const double delta )
{
  char* proc = "fff_clustering_FCM"; 
  int fd = X->size2;      /* feature dimension */
  int k = Centers->size1; /* number of clusters */
  int i,j,l;
  double normdC,normC; 
  double dx,J;
  int verbose = 0;

  fff_matrix* U = fff_matrix_new(X->size1, Centers->size1);
  fff_matrix* Centers_old = fff_matrix_new(Centers->size1, Centers->size2);
  
  fff_matrix_set_all( Centers_old,0);
  _fff_FCM_init(U);
  _fff_fuzzy_Estep(Centers,X,U);
  
  for (l=0 ; l<maxiter ; l++){
    _fff_fuzzy_Mstep(U,X,Centers);
    _fff_fuzzy_Estep(Centers,X,U); 
    J  = _fff_FCM_functional(X, Centers, U);
    
    if (verbose)
      printf("%s Iter %d functional J = %f \n",proc,l,J);
    
    normdC = 0;
    normC = 0;
    for (i=0 ; i<k ; i++)  
      for (j=0 ; j<fd ; j++){
	dx = fff_matrix_get(Centers_old,i,j)-fff_matrix_get(Centers,i,j);
	normdC += dx*dx;
	dx = fff_matrix_get(Centers_old,i,j);
	normC += dx*dx;
      }
    normdC = sqrt(normdC);
    normC = sqrt(normC);
    
    fff_matrix_memcpy(Centers_old, Centers);
    if (normdC<delta*normC) break;
    
  } 
  /* compute "hard" memberships*/
  _fff_Mstep(Label,X,Centers);
  
  fff_matrix_delete(Centers_old);
  fff_matrix_delete(U);
  return(0);
}

/* Initialization of the Centers matrix by picking radomly points from X */ 
static void _fff_FCM_init(fff_matrix* U)
{
  int C = U->size2;
  int N = U->size1;
  int index,i,n;
  rk_state state; 
  
  for (i=0; i<N*C; i++) 
    U->data[i] = (1-sqrt(2)/2)/C;
  
  rk_seed(1, &state);
  
  for (n=0 ; n<N ; n++){
    index = (int)(C*rk_double(&state)); 
    U->data[n*C+index]+= sqrt(2)/2; 
  }

}

/* functional of the FCM algorithm */
static double _fff_FCM_functional(const fff_matrix* X, const fff_matrix* Centers, const fff_matrix *U)
{
  int n,c,t;
  double dx,J,lu;
  int N = X->size1;
  int T = X->size2;
  int C = Centers->size1;

  J=0;
  for (n=0; n<N; n++){
    for (c=0; c<C; c++){
      lu = fff_matrix_get(U,n,c);
          
      for ( t=0; t<T; t++){
        dx = fff_matrix_get(X,n,t)-fff_matrix_get(Centers,c,t);
        J += lu*lu*dx*dx;
      }
    }
  }
  return(J);
}

/* Mstep of the FCM algo: compute fuzzy memberships */
static void _fff_fuzzy_Mstep ( fff_matrix* U, const fff_matrix* X, const fff_matrix* Centers)
{
  double dx,mindist,auxdist,temp;
   int index;
   int n,t,c;
   int N = X->size1;
   int T  = X->size2;
   int C = Centers->size1;

   fff_vector* dist =  fff_vector_new(C);  

   for (n=0 ; n<N ; n++){
     mindist = 1.0;
     index = 0;
     /* Compute the distances */
     for (c=0 ; c<C ; c++){
       auxdist = 0;
       for ( t=0 ; t<T ; t++){
	 dx = fff_matrix_get(X,n,t)-fff_matrix_get(Centers,c,t);
	 auxdist += dx*dx;
       }
       fff_vector_set(dist,c,auxdist);
       if (auxdist==0){
	 mindist = 0; 
	 index = c;
       }
     }
     /* Update the memberships */ 
     if (mindist==0){
       for (c=0 ; c<C ; c++)
	 fff_matrix_set(U,n,c,0);
       fff_matrix_set(U,n,index,1);
     }
     else{ 
       auxdist = 0;
       for (c=0; c<C; c++)
	 auxdist += 1.0/fff_vector_get(dist,c);
       for (c=0; c<C; c++){
	 temp = 1.0/(fff_vector_get(dist,c) * auxdist);
	 fff_matrix_set(U,n,c,temp);
       }
     }  
   }
   fff_vector_delete(dist);
}

/* E step of the FCM algo: update the cluster centers */
static void _fff_fuzzy_Estep(fff_matrix* Centers, const fff_matrix* X, const fff_matrix* U)
{
  int C = (int)(Centers->size1);
  int c;
  fff_vector *count = fff_vector_new(C);
  fff_vector *v = fff_vector_new(Centers->size2);
  fff_matrix* Uc = fff_matrix_new(U->size1, U->size2);
  fff_vector *aux;

  fff_matrix_memcpy (Uc, U);
  fff_matrix_mul_elements(Uc,Uc);

  fff_blas_dgemm (CblasTrans, CblasNoTrans,1,Uc,X, 0, Centers);

  fff_vector_set_all(count,0);
  aux = fff_vector_new(U->size1);
  fff_vector_set_all(aux,1.);
  
  fff_blas_dgemv (CblasTrans, 1, Uc, aux, 0, count);
  
  for (c=0 ; c<C ; c++){
    if (fff_vector_get(count,c)>0){
      fff_matrix_get_row (v,Centers,c);
      fff_vector_scale (v, 1./fff_vector_get(count,c));
      fff_matrix_set_row (Centers,c,v);
    }
  }
 
  fff_vector_delete(count);
  fff_vector_delete(aux);
  fff_vector_delete(v);
  fff_matrix_delete(Uc);
}


int fff_clustering_OntoLabel(const fff_array * Label, const long k)
{
  
  char* proc = "_fff_clustering_OntoLabel";
  int bverbose = 0;
  int i,n = Label->dimX;
  double mL,ML;
  double * cLabel; 

  fff_array_extrema ( &mL, &ML, Label );

  if (mL != 0){
    if (bverbose)
      printf("%s Inconsistant Labelling mL= %d \n",proc,(int)mL); 
    return(0);
  }
  if (ML != k-1){
    if (bverbose)
      printf("%s Inconsistant Labelling ML = %d \n",proc,(int)ML);
    
    return(0);
  }
  
  cLabel = (double *) calloc(Label->dimX, sizeof(double));
  for (i=0 ; i<n ; i++) cLabel[i] = (double)fff_array_get1d(Label,i);
  sort_ascending(cLabel,n);
   for (i=1 ; i<n ; i++)
    if (cLabel[i]>cLabel[i-1])
      if (cLabel[i]!= cLabel[i-1]+1)
		{
		  if (bverbose)
			printf("%s, Inconsistant Labelling i=%d \n",proc,(int)(cLabel[i-1]+1));
		  
		  return(0);
		}
   free(cLabel);
  
return(1);
}

