#include "fff_BPmatch.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


static int  _fff_swapindex(fff_array *idx, const fff_graph *G);

static int _fff_matrix_normalize_rows(fff_matrix* A);



static int  _fff_swapindex(fff_array *idx, const fff_graph *G)
{
  int V = G->V; 
  int e,i, E = G->E;

  fff_array *temp = fff_array_new1d(FFF_LONG,V*V);
  fff_array_set_all(temp,-1);
  for (e=0 ; e<E ; e++){
	i = G->eA[e]*V+G->eB[e];
	fff_array_set1d(temp,i,e);
  }
  for (e=0 ; e<E ; e++){
	i = G->eB[e]*V+G->eA[e];
	fff_array_set1d(idx,e,fff_array_get1d(temp,i));
  }
  fff_array_delete(temp);
  return(0); 
}

static int _fff_matrix_normalize_rows(fff_matrix* A)
{
  /* normalize all the rows to sum 1 -when possible*/
  int i;
  double sv;
  
  fff_vector v;
  fff_vector* ones = fff_vector_new(A->size2);
  fff_vector_set_all(ones,1);
  for (i=0 ; i<A->size1 ; i++){
	v = fff_matrix_row (A, i);
	sv = fff_blas_ddot (&v, ones);
	if (sv != 0)
	  fff_vector_scale(&v,1./sv);
  }
  fff_vector_delete(ones);

  return(0);
}

extern int fff_BPmatch(fff_matrix * source, fff_matrix * target, fff_matrix * adjacency, fff_matrix * belief, double d0)
{
  int i,j,k,e,A,B;
  double dx,dist;
  double dB,b=0;
  double eps = 1.e-12;
  int maxiter = 20;
  int verbose = 0;
  
  /* Some basic checks */
  
  int n1 = source->size1;
  int n2 = target->size1;
  int p = source->size2;
  double sqs = 2*d0*d0;
  double dth = 4.5*sqs;

  fff_vector wi,vi; 
  fff_graph *G;
  int E; 
  fff_array *tag; 
  fff_matrix **T;
  fff_vector *u, *v, *v1, *v2;
  fff_matrix *Msg, *PMsg, *old_belief, *initial_belief;
  
  if (source->size2 != target->size2){
	FFF_WARNING("Incompaticle dimension four source and target\n");
	return (0);
  }
  if (adjacency->size1 != adjacency->size2){
	FFF_WARNING("adjacency is not square\n");
	return (0);
  }
  if (adjacency->size1 != n1){
	FFF_WARNING("Bad size for adjacency \n");
	return (0);
  }
  if (belief->size1 != n1){
	FFF_WARNING("Bad size for belief\n");
	return (0);
  }
  if (belief->size2 != n2){
	FFF_WARNING("Bad size for belief\n");
	return (0);
  }
  fff_matrix_set_all(belief,0);
  
  /* Initialization of the probabilistic correspondences  */
  for (i=0 ; i<n1 ; i++ ){
	for (j=0 ; j<n2 ; j++){
	  dist = 0;
	  for ( k=0 ; k<p ; k++){
		dx = fff_matrix_get(source,i,k)-fff_matrix_get(target,j,k);
		dist += dx*dx;
		if (dist>dth) break;
	  }
	  if (dist<dth)
		b = exp(-dist/sqs);
	  else
		b=0;
	  fff_matrix_set(belief,i,j,b);

	}
  }
  _fff_matrix_normalize_rows(belief);
  
  /* Initialization of the transition matrices */
  vi = fff_matrix_diag(adjacency);
  fff_vector_set_all(&vi,0);
  
  fff_matrix_to_graph(&G, adjacency);
  fff_remove_null_edges(&G);
    
  if (G->E==0)
	return(0);
  E = G->E;
  
  tag = fff_array_new1d(FFF_LONG,E);
  _fff_swapindex(tag,G);
  
  T = (fff_matrix **) calloc(E,sizeof(fff_matrix*)); 
  u = fff_vector_new(p);
  v = fff_vector_new(p);
  
  for (e=0 ; e<E ; e++){
	A = G->eA[e];
	B = G->eB[e];
	T[e] = fff_matrix_new(n2,n2);
	
	vi = fff_matrix_row (source,A);
	fff_vector_memcpy(u,&vi);
	vi = fff_matrix_row(source,B);
	fff_vector_sub(u,&vi);
	
	for (i=0 ; i<n2 ; i++){
	  for (j=0 ; j<n2 ; j++){
		fff_vector_memcpy(v,u);	
		vi = fff_matrix_row(target,i);
		fff_vector_sub(v,&vi);
		vi = fff_matrix_row(target,j);
		fff_vector_add(v,&vi);
		dist = fff_blas_ddot (v,v);
		b = exp(-dist/sqs);
		fff_matrix_set(T[e],i,j,b);
	  }
	}
	_fff_matrix_normalize_rows(T[e]);
  }
  
  /* Initialization of the messages */
  Msg = fff_matrix_new(E,n2);
  PMsg = fff_matrix_new(E,n2);
  
  for (e=0 ; e<E ; e++){
	int A = G->eA[e];
	vi = fff_matrix_row (PMsg,e);
	wi = fff_matrix_row (belief,A);
	fff_vector_memcpy (&vi, &wi);
  } 

  /* message passing algorithm */
  old_belief = fff_matrix_new(n1,n2);
  initial_belief = fff_matrix_new(n1,n2);
  fff_matrix_memcpy(initial_belief, belief);
  v1 = fff_vector_new(n2);
  v2 = fff_vector_new(n2);
  
  for (i=0 ; i<maxiter ; i++){

	fff_matrix_memcpy(old_belief, belief);
	fff_matrix_memcpy(belief, initial_belief);
	fff_matrix_memcpy(Msg, PMsg);

	/* compute the msgs  */
	for (e=0 ; e<E ; e++){
	  vi = fff_matrix_row (Msg,e);	
	  fff_matrix_get_row(v1,Msg,e);
	  fff_blas_dgemv (CblasTrans, 1., T[e], v1, 0.,&vi); 
	}

	_fff_matrix_normalize_rows(Msg);

	/* update the beliefs */
	for (e=0 ; e<E ; e++){
	  B = G->eB[e];
	  wi = fff_matrix_row (belief,B);
	  vi = fff_matrix_row (Msg,e);
	  fff_vector_mul (&wi,&vi );
	}
	_fff_matrix_normalize_rows(belief);

	/* stopping criterion */
	fff_matrix_sub (old_belief, belief);	
	fff_matrix_mul_elements (old_belief,old_belief );
	dB = fff_matrix_sum(old_belief);

	if (dB<eps){
	  if (verbose)
		printf("iter %d, diff %f %f \n",i,dB,eps);
	  break;
	}

	/* Prepare the next messages */
	for (e=0 ; e<E ; e++){
	  fff_matrix_get_row (v1,Msg,fff_array_get1d(tag,e));
	  for (j=0 ; j<n2 ;j++) 
		if (fff_vector_get(v1,j)<eps) 
		  fff_vector_set(v1,j,eps);
	  fff_matrix_get_row (v2,belief, G->eA[e]);
	  fff_vector_div(v2,v1);
	  fff_matrix_set_row (PMsg, e, v2);
	}		
  } 
  
  /* Final steps */
  _fff_matrix_normalize_rows(belief);
    
  fff_vector_delete(v2);
  fff_vector_delete(v1);
  fff_matrix_delete(old_belief);
  fff_matrix_delete(initial_belief);
  
  fff_matrix_delete(Msg);
  fff_matrix_delete(PMsg); 
  
  for (e=0 ; e<E ; e++)
	fff_matrix_delete(T[e]);
  
  free(T);
  
  fff_vector_delete(u);
  fff_vector_delete(v);
 
  fff_array_delete(tag);  
  fff_graph_delete(G);
  
  return(1);
  
}
