#include "fff_field.h"
#include "fff_routines.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>


static long  _fff_list_add( long *listn, double *listd,  const long newn, const double newd, const long k, const long j);


static long _fff_list_move( long *listn, double *listd,  const long newn, const double newd, const long k, const long j);

static int _fff_field_maxima_rth(fff_array *depth, const fff_graph* G, const fff_vector *field, const int rec, const double th);

/******************************************************/
/**************** ancillary stuff *********************/
/******************************************************/

/**********************************************************************
 *************************** Field Maxima ******************************
**********************************************************************/

extern int fff_field_maxima(fff_array *depth, const fff_graph* G, const fff_vector *field)
{
  int N = G->V;
  int k = fff_field_maxima_r(depth, G, field, N);
  return(k);
}
  
extern int fff_field_get_maxima(fff_array **depth, fff_array ** idx,const fff_graph* G, const fff_vector *field)
{
  int i,j;
  int N = field->size;
  fff_array* ldepth = fff_array_new1d(FFF_LONG,N);
  fff_array *de, *id;

  int q = fff_field_maxima(ldepth, G, field);
  if (q==0) 
    return(0);
  de = fff_array_new1d(FFF_LONG,q);
  id = fff_array_new1d(FFF_LONG,q);
  
  j = 0;
  for(i=0 ; i<N ; i++)
    if (fff_array_get1d(ldepth,i)>0){
      fff_array_set1d(de,j,fff_array_get1d(ldepth,i));
      fff_array_set1d(id,j,i);
      j++;
    }

  *depth = de;
  *idx = id;
  fff_array_delete(ldepth);
  return(q);
}

extern int fff_field_get_maxima_th(fff_array **depth, fff_array ** idx,const fff_graph* G, const fff_vector *field, const double th)
{
  int i,j;
  int N = field->size;
  fff_array* ldepth = fff_array_new1d(FFF_LONG,N);
  fff_array *de, *id; 

  int q = _fff_field_maxima_rth(ldepth, G, field, N,th);
  if (q==0) 
    return(0);
  de = fff_array_new1d(FFF_LONG,q);
  id = fff_array_new1d(FFF_LONG,q);
  
  j = 0;
  for(i=0 ; i<N ; i++)
    if (fff_array_get1d(ldepth,i)>0){
      fff_array_set1d(de,j,fff_array_get1d(ldepth,i));
      fff_array_set1d(id,j,i);
      j++;
    }

  *depth = de;
  *idx = id;
  fff_array_delete(ldepth);
  return(q);
}

extern int fff_field_maxima_r(fff_array *depth, const fff_graph* G, const fff_vector *field, const int rec)
{
  int i,r,N = G->V;
  int E = G->E;
  int nA,nB,remain;
  double delta;
  fff_array *win; 
  fff_vector *mfield, *Mfield; 
  int k; 

  if (((int)(field->size) != N)||((int)(depth->dimX) !=N)){
    FFF_WARNING("Size pof the graph and of the vectors do not match");
    return(0);
  }

  win = fff_array_new1d(FFF_LONG,N);
  mfield = fff_vector_new(N);
  Mfield = fff_vector_new(N);
  if (!mfield)    return(0);
  if (!Mfield)    return(0);	
  if (!win)    return(0);

  fff_vector_memcpy(mfield, field);
  fff_vector_memcpy(Mfield, field);
  fff_array_set_all(win,1);
  fff_array_set_all(depth,0);
   
   /* Iterative dilatation */
  for (r=0 ; r<rec ; r++){
    for (i=0 ; i<E ; i++){
      nA = G->eA[i];
      nB = G->eB[i];
      if (fff_vector_get(mfield,nA)<fff_vector_get(mfield,nB)){
	fff_array_set1d (win, nA, 0);
	if (fff_vector_get(Mfield,nA)<fff_vector_get(mfield,nB))
	  fff_vector_set (Mfield,nA, fff_vector_get(mfield,nB));	
      }
    }
    remain = 0;
    
    fff_vector_sub(mfield,Mfield);
    delta = fff_blas_ddot (mfield,mfield);
    fff_vector_memcpy(mfield, Mfield);
    fff_array_add(depth,win);
    for (i=0 ; i<N ; i++)
      remain += (fff_array_get1d(win,i)>0);
    if (remain<2)
      break;
    if (delta==0)
      break; 
    /* stop when all the maxima have been found */
  }

  k = 0;
  for (i=0 ; i<N ; i++)
    k+= (fff_array_get1d(depth,i)>0);

  fff_array_delete(win);
  fff_vector_delete(mfield);
  fff_vector_delete(Mfield);

  return(k);
}

static int _fff_field_maxima_rth(fff_array *depth, const fff_graph* G, const fff_vector *field, const int rec, const double th)
{
  int i,r,N = G->V;
  int E = G->E;
  int nA,nB,remain;
  double delta;
  fff_array *win; 
  fff_vector *mfield, *Mfield; 
  int k; 

  if (((int)(field->size) != N)||((int)(depth->dimX) !=N)){
    FFF_WARNING("Size pof the graph and of the vectors do not match");
    return(0);
  }

  win = fff_array_new1d(FFF_LONG,N);
  mfield = fff_vector_new(N);
  Mfield = fff_vector_new(N);
  if (!mfield)    return(0);
  if (!Mfield)    return(0);	
  if (!win)    return(0);

  fff_vector_memcpy(mfield, field);
  fff_vector_memcpy(Mfield, field);
  fff_array_set_all(win,0);
  fff_array_set_all(depth,0);
   
  for (i=0 ; i<N ; i++)
    if (fff_vector_get(field,i)>th)
      fff_array_set1d(win,i,1);

   /* Iterative dilatation */
  for (r=0 ; r<rec ; r++){
    for (i=0 ; i<E ; i++){
      nA = G->eA[i];
      nB = G->eB[i];
      if (fff_vector_get(mfield,nA)<fff_vector_get(mfield,nB)){
	fff_array_set1d (win, nA, 0);
	if (fff_vector_get(Mfield,nA)<fff_vector_get(mfield,nB))
	  fff_vector_set (Mfield,nA, fff_vector_get(mfield,nB));	
      }
    }
    remain = 0;
    
    fff_vector_sub(mfield,Mfield);
    delta = fff_blas_ddot (mfield,mfield);
    fff_vector_memcpy(mfield, Mfield);
    fff_array_add(depth,win);
    for (i=0 ; i<N ; i++)
      remain += (fff_array_get1d(win,i)>0);
    if (remain<2)
      break;
    if (delta==0)
      break; 
    /* stop when all the maxima have been found */
  }

  k = 0;
  for (i=0 ; i<N ; i++)
    k+= (fff_array_get1d(depth,i)>0);

  fff_array_delete(win);
  fff_vector_delete(mfield);
  fff_vector_delete(Mfield);

  return(k);
}

/**********************************************************************
 *************************** Field Minima ******************************
**********************************************************************/

extern int fff_field_minima(fff_array *depth, const fff_graph* G, const fff_vector *field)
{
  int N = G->V;
  int k = fff_field_minima_r(depth, G, field, N);
  return(k);
}
  
extern int fff_field_get_minima(fff_array **depth, fff_array ** idx,const fff_graph* G, const fff_vector *field)
{
  int i,j;
  int N = field->size;
  fff_array* ldepth = fff_array_new1d(FFF_LONG,N);
  fff_array *de, *id; 

  int q = fff_field_minima(ldepth, G, field);
  if (q==0) 
    return(0);
  de = fff_array_new1d(FFF_LONG,q);
  id = fff_array_new1d(FFF_LONG,q);
  
  j = 0;
  for(i=0 ; i<N ; i++)
    if (fff_array_get1d(ldepth,i)>0){
      fff_array_set1d(de,j,fff_array_get1d(ldepth,i));
      fff_array_set1d(id,j,i);
      j++;
    }

  *depth = de;
  *idx = id;
  fff_array_delete(ldepth);
  return(q);
}

extern int fff_field_minima_r(fff_array *depth, const fff_graph* G, const fff_vector *field, const int rec)
{
  int i,r,N = G->V;
  int E = G->E;
  int nA,nB,remain;
  double delta;
  fff_array *win; 
  fff_vector *mfield, *Mfield; 
  int k; 

  if (((int)(field->size) != N)||((int)(depth->dimX) !=N)){
     FFF_WARNING("Size pof the graph and of the vectors do not match");
     return(0);
  }

  win = fff_array_new1d(FFF_LONG,N);
  mfield = fff_vector_new(N);
  Mfield = fff_vector_new(N);
  if (!mfield)    return(0);
  if (!Mfield)    return(0);	
  if (!win)    return(0);

  fff_vector_memcpy(mfield, field);
  fff_vector_memcpy(Mfield, field);
  fff_array_set_all(win,1);
  fff_array_set_all(depth,0);
   
   /* Iterative dilatation */
  for (r=0 ; r<rec ; r++){
    for (i=0 ; i<E ; i++){
      nA = G->eA[i];
      nB = G->eB[i];
      if (fff_vector_get(mfield,nA)>fff_vector_get(mfield,nB)){
	fff_array_set1d (win, nA, 0);
	if (fff_vector_get(Mfield,nA)>fff_vector_get(mfield,nB))
	  fff_vector_set (Mfield,nA, fff_vector_get(mfield,nB));	
      }
    }
    remain = 0;
    
    fff_vector_sub(mfield,Mfield);
    delta = fff_blas_ddot (mfield,mfield);
    fff_vector_memcpy(mfield, Mfield);
    fff_array_add(depth,win);
    for (i=0 ; i<N ; i++)
      remain += (fff_array_get1d(win,i)>0);
    if (remain<2)
      break;
    if (delta==0)
      break; 
    /* stop when all the minima have been found */
  }

  k = 0;
  for (i=0 ; i<N ; i++)
    k+= (fff_array_get1d(depth,i)>0);

  fff_array_delete(win);
  fff_vector_delete(mfield);
  fff_vector_delete(Mfield);

  return(k);
}


/************************************************************************/
/************* Diffusion ************************************************/
/************************************************************************/



int fff_field_diffusion( fff_vector *field, const fff_graph* G)
{
    int V = G->V;
    int E = G->E;
    int i;
    double temp;
    fff_vector *cfield; 

    if ((int)(field->size)!=V){
      FFF_WARNING(" incompatible matrix size \n");
      return(1);
    }

    cfield = fff_vector_new(V);
    fff_vector_memcpy(cfield,field);
    fff_vector_set_all(field,0);

    for(i=0 ; i<E ; i++){ 
      temp = fff_vector_get(field,G->eB[i])+ G->eD[i]*fff_vector_get(cfield,G->eA[i]);
      fff_vector_set(field,G->eB[i],temp);	  
    }
   
    fff_vector_delete(cfield);
    return(0);
}

int fff_field_md_diffusion( fff_matrix *field, const fff_graph* G)
{
    int E = G->E;
    int V = G->V;
    int i,nc,nr;    
    fff_matrix *cfield;
    fff_vector vi, *v; 

    nc = field->size2;
    nr = field->size1;

    if (nr!=V){
      FFF_WARNING(" incompatible matrix size \n");
      return(1);
    }
    
    cfield = fff_matrix_new(nr,nc);
    fff_matrix_memcpy(cfield,field);
    fff_matrix_set_all(field,0);
    v = fff_vector_new(nc);
	/* */
	/* */
    for(i=0 ; i<E ; i++){
	  vi = fff_matrix_row (field, G->eB[i]);
	  fff_matrix_get_row (v,cfield, G->eA[i]);
	  fff_vector_scale(v,G->eD[i]);
	  fff_vector_add(&vi,v );
    }
	fff_vector_delete(v);
    
    fff_matrix_delete(cfield);
    return(0);
}


/************************************************************************/
/************* Mathematical morphology **********************************/
/************************************************************************/


extern int fff_field_dilation(fff_vector *field, const fff_graph* G, const int rec)
{
  int i,r,N = G->V;
  int E = G->E;
  int nA,nB;
  int ri = 0;
  fff_vector* mfield; 

  if ((int)(field->size) != N){
     FFF_WARNING("Size pof the graph and of the vectors do not match");
     return(0);
  }

  mfield = fff_vector_new(N);
  if (!mfield)    return(0);
  
  /* Iterative dilatation */
  for (r=0 ; r<rec ; r++){    
    fff_vector_memcpy(mfield,field);
    for (i=0 ; i<E ; i++){
      nA = G->eA[i];
      nB = G->eB[i];
      if (fff_vector_get(field,nA)<fff_vector_get(mfield,nB))
	fff_vector_set (field,nA, fff_vector_get(mfield,nB));	
    }
  }
  fff_vector_delete(mfield);
  
  return(ri);
}

extern int fff_field_erosion(fff_vector *field, const fff_graph* G, const int rec)
{
  int i,r,N = G->V;
  int E = G->E;
  int nA,nB;
  int ri = 0;
  fff_vector* mfield;   

  if ((int)(field->size) != N){
     FFF_WARNING("Size pof the graph and of the vectors do not match");
     return(0);
  }

  mfield = fff_vector_new(N);
  if (!mfield)    return(0);
  
  /* Iterative dilatation */
  for (r=0 ; r<rec ; r++){    
    fff_vector_memcpy(mfield,field);
    for (i=0 ; i<E ; i++){
      nA = G->eA[i];
      nB = G->eB[i];
      if (fff_vector_get(field,nA)>fff_vector_get(mfield,nB))
	fff_vector_set (field,nA, fff_vector_get(mfield,nB));	
    }
  }
  fff_vector_delete(mfield);
  
  return(ri);
}

extern int fff_field_opening(fff_vector *field, const fff_graph* G, const int rec)
{
  int ri = 0;
  fff_field_erosion(field,G,rec);
  fff_field_dilation(field,G,rec);
  return(ri);
}

extern int fff_field_closing(fff_vector *field, const fff_graph* G, const int rec)
{
  int ri = 0;
  fff_field_dilation(field,G,rec);
  fff_field_erosion(field,G,rec);
  return(ri);
}


extern int fff_custom_watershed(fff_array **idx, fff_array **depth, fff_array **major, fff_array* label,  const fff_vector *field, const fff_graph* G)
{
  int i,r,N = G->V;
  int E = G->E;
  int nA,nB,remain;
  double delta;
  int k; 
  fff_array *win, *maj1, *maj2, *incwin; 
  fff_vector *mfield, *Mfield; 
  int j,aux;
  fff_array *lidx, *ldepth, *lmajor;

  if ((int)(field->size) != N){
     FFF_WARNING("Size pof the graph and of the vectors do not match");
     return(0);
  } 
  k = 0;
  
  win = fff_array_new1d(FFF_LONG,N);
  maj1 = fff_array_new1d(FFF_LONG,N);
  maj2 = fff_array_new1d(FFF_LONG,N);
  incwin = fff_array_new1d(FFF_LONG,N);
  mfield = fff_vector_new(N);
  Mfield = fff_vector_new(N);
  if (!mfield) return(0);
  if (!Mfield) return(0);	
  if (!win) return(0);
  
  fff_vector_memcpy(mfield, field);
  fff_vector_memcpy(Mfield, field);
  fff_array_set_all(win,1);
  fff_array_set_all(incwin,0);
  
  for (i=0 ; i<N ; i++)
    fff_array_set1d(maj1,i,i);
  fff_array_copy(maj2, maj1);
   
   /* Iterative dilatation  */
  for (r=0 ; r<N ; r++){
    for (i=0 ; i<E ; i++){
      nA = G->eA[i];
      nB = G->eB[i];
      if (fff_vector_get(mfield,nA)<fff_vector_get(mfield,nB)){
	fff_array_set1d (win, nA, 0);
	if (fff_vector_get(Mfield,nA)<fff_vector_get(mfield,nB)){
	  fff_vector_set (Mfield,nA, fff_vector_get(mfield,nB));
	  fff_array_set1d(maj2,nA,fff_array_get1d( maj2,nB));
	  if (fff_array_get1d(incwin,nA)==r)
	    fff_array_set1d(maj1,nA,fff_array_get1d(maj2,nB));
	}
      }
    }
    remain = 0;
    
    fff_vector_sub(mfield,Mfield);
    delta = fff_blas_ddot (mfield,mfield);
    fff_vector_memcpy(mfield, Mfield);
    fff_array_add(incwin,win);
    for (i=0 ; i<N ; i++)
      remain += (fff_array_get1d(win,i)>0);
    
    if (remain<2)
      break;
    if (delta==0)
      break; 
    /* stop when all the maxima have been found  */
  }
  
  /* get the local maximum associated with any point  */
  for (i=0 ; i<N ; i++){
    j = fff_array_get1d(maj1,i);
    while (fff_array_get1d(incwin,j)==0)
      j = fff_array_get1d(maj1,j);
    fff_array_set1d(maj1,i,j);
  }

  /* number of bassins  */
  for (i=0 ; i<N ; i++)
    k+= (fff_array_get1d(incwin,i)>0);

  lidx = fff_array_new1d(FFF_LONG,k);
  ldepth = fff_array_new1d(FFF_LONG,k);
  lmajor = fff_array_new1d(FFF_LONG,k); 

  /* write the maxima and related stuff  */
  j=0;
  for (i=0 ; i<N ; i++)
    if (fff_array_get1d(incwin,i)>0){
      fff_array_set1d(lidx,j,i);
      fff_array_set1d(ldepth,j, fff_array_get1d(incwin,i));
      fff_array_set1d(maj2,i,j);/* ugly, but OK  */
      j++;
    }
  for (j=0 ; j<k ; j++){
    i = fff_array_get1d(lidx,j);
    if (fff_array_get1d(maj1,i) != i){ /* i is not a global maximum */
      aux = fff_array_get1d(maj2,fff_array_get1d(maj1,i));
      fff_array_set1d(lmajor,j,aux);
    }
  else
    fff_array_set1d(lmajor,j,j);
  }
  
  /* Finally set the labels */
  for (i=0 ; i<N ; i++){
    aux = fff_array_get1d(maj2,fff_array_get1d(maj1,i));
    fff_array_set1d(label,i,aux);
  }
  for (j=0 ; j<k ; j++){
    i = fff_array_get1d(lidx,j);
    fff_array_set1d(label,i,j);
  }
  
  *idx = lidx;
  *depth = ldepth;
  *major = lmajor;

  fff_array_delete(win);
  fff_array_delete(maj1);
  fff_array_delete(maj2);
  fff_array_delete(incwin);
  fff_vector_delete(mfield);
  fff_vector_delete(Mfield);

  return(k);
}

extern int fff_custom_watershed_th(fff_array **idx, fff_array **depth, fff_array **major, fff_array* label,  const fff_vector *field, const fff_graph* G, const double th)
{
  int i,r,N = G->V;
  int E = G->E;
  int nA,nB,remain;
  double delta;
  int k; 
  fff_array *win, *maj1, *maj2, *incwin; 
  fff_vector *mfield, *Mfield; 
  int j, aux;
  fff_array *lidx, *ldepth, *lmajor;
  

  if ((int)(field->size) != N){
    FFF_WARNING("Size pof the graph and of the vectors do not match");
    return(0);
  } 
  k = 0;
  
  win = fff_array_new1d(FFF_LONG,N);
  maj1 = fff_array_new1d(FFF_LONG,N);
  maj2 = fff_array_new1d(FFF_LONG,N);
  incwin = fff_array_new1d(FFF_LONG,N);
  mfield = fff_vector_new(N);
  Mfield = fff_vector_new(N);
  if (!mfield) return(0);
  if (!Mfield) return(0);	
  if (!win) return(0);

  fff_vector_memcpy(mfield, field);
  fff_vector_memcpy(Mfield, field);
  fff_array_set_all(win,0);
  fff_array_set_all(incwin,0);
  
  for (i=0 ; i<N ; i++){
    fff_array_set1d(maj1,i,i);
    if (fff_vector_get(field,i)>th)
      fff_array_set1d(win,i,1);
  }
  fff_array_copy(maj2, maj1);

   /* Iterative dilatation  */
  for (r=0 ; r<N ; r++){
    for (i=0 ; i<E ; i++){
      nA = G->eA[i];
      nB = G->eB[i];
      if (fff_vector_get(field,nA)>th)
		if (fff_vector_get(mfield,nA)<fff_vector_get(mfield,nB)){
		  fff_array_set1d (win, nA, 0);
		  if (fff_vector_get(Mfield,nA)<fff_vector_get(mfield,nB)){
			fff_vector_set (Mfield,nA, fff_vector_get(mfield,nB));
			fff_array_set1d(maj2,nA,fff_array_get1d( maj2,nB));
			if (fff_array_get1d(incwin,nA)==r)
	      fff_array_set1d(maj1,nA,fff_array_get1d(maj2,nB));
		  }
		}
    }
    remain = 0;
    
    fff_vector_sub(mfield,Mfield);
    delta = fff_blas_ddot (mfield,mfield);
    fff_vector_memcpy(mfield, Mfield);
    fff_array_add(incwin,win);
    for (i=0 ; i<N ; i++)
      remain += (fff_array_get1d(win,i)>0);
    
    if (remain<2)
      break;
    if (delta==0)
      break; 
    /* stop when all the maxima have been found  */
  }
  
  /* get the local maximum associated with any point  */
  for (i=0 ; i<N ; i++){
    if (fff_vector_get(field,i)>th){
      j = fff_array_get1d(maj1,i);
      while (fff_array_get1d(incwin,j)==0)
	j = fff_array_get1d(maj1,j);
      fff_array_set1d(maj1,i,j);
    }
  }

  /* number of bassins  */
  for (i=0 ; i<N ; i++)
    k+= (fff_array_get1d(incwin,i)>0);

  if (k<1){
    lidx = NULL;
    ldepth = NULL;
    lmajor = NULL;
    fff_array_set_all(label,-1);
  }
  else{
    lidx = fff_array_new1d(FFF_LONG,k);
    ldepth = fff_array_new1d(FFF_LONG,k);
    lmajor = fff_array_new1d(FFF_LONG,k); 
    
    /* write the maxima and related stuff  */
    j=0;
    for (i=0 ; i<N ; i++)
      if (fff_array_get1d(incwin,i)>0){
	fff_array_set1d(lidx,j,i);
	fff_array_set1d(ldepth,j, fff_array_get1d(incwin,i));
	fff_array_set1d(maj2,i,j);/* ugly, but OK  */
	j++;
      }
    for (j=0 ; j<k ; j++){
      i = fff_array_get1d(lidx,j);
      if (fff_array_get1d(maj1,i) != i){ /* i is not a global maximum */
	aux = fff_array_get1d(maj2,fff_array_get1d(maj1,i));
	fff_array_set1d(lmajor,j,aux);
      }
      else
	fff_array_set1d(lmajor,j,j);
    }
    
    /* Finally set the labels */
    for (i=0 ; i<N ; i++){
      if (fff_vector_get(field,i)<th)
	fff_array_set1d(label,i,-1);
      else{
	aux = fff_array_get1d(maj2,fff_array_get1d(maj1,i));
	fff_array_set1d(label,i,aux);
      }
    }
    for (j=0 ; j<k ; j++){
      i = fff_array_get1d(lidx,j);
      fff_array_set1d(label,i,j);
    }
  }  
  *idx = lidx;
  *depth = ldepth;
  *major = lmajor;
  
  fff_array_delete(win);
  fff_array_delete(maj1);
  fff_array_delete(maj2);
  fff_array_delete(incwin);
  fff_vector_delete(mfield);
  fff_vector_delete(Mfield);

  return(k);
}




extern long fff_field_bifurcations(fff_array **Idx, fff_vector **Height, fff_array **Father, fff_array* label,  const fff_vector *field, const fff_graph* G, const double th)
{
  long i,j,k,l,win,start, end;
  long V = G->V;
  long E = G->E;
  long ri = 0;
  long ll = 0;
  fff_array *cindices, *neighb;
  fff_vector *weight; 
  fff_vector *cfield;
  fff_array *father, *possible, *idx; 
  fff_vector *height; 
  fff_array* papa;
  fff_array* indices;
  fff_vector* hauteur;
  long *p; 
  long q; 

  /* argument checking */
  if ((label->dimX)!=V){
    FFF_WARNING(" incompatible matrix size \n");
    return(1);
  }

  /* initializations */
  cindices = fff_array_new1d(FFF_LONG,V+1);
  neighb = fff_array_new1d(FFF_LONG,E);
  weight = fff_vector_new(E);
  
  ri = fff_graph_to_neighb(cindices,neighb,weight,G);
  
  /* sort the data */
  cfield = fff_vector_new(V);
  fff_vector_memcpy(cfield,field);
  fff_vector_scale (cfield, -1);
  p = (long *) calloc(V,sizeof(long));
  sort_ascending_and_get_permutation( cfield->data, p, cfield->size );
  fff_vector_delete(cfield);
  

  fff_array_set_all(label,-1);
  father = fff_array_new1d(FFF_LONG,2*V);
  for (i=0; i<2*V ; i++)
	fff_array_set1d(father,i,i);
  
  possible = fff_array_new1d(FFF_LONG,V);
  idx = fff_array_new1d(FFF_LONG,2*V);
  height = fff_vector_new(2*V);

  for (i=0; i<V ; i++){
	win = p[i];
	if (fff_vector_get(field,win)<th) break;
	else{
	  start = fff_array_get1d(cindices,win);
	  end = fff_array_get1d(cindices,win+1);
	  fff_array_set_all(possible,-1);
	  q = 0;
	  
	  for (j=start ; j<end ; j++){
		k = fff_array_get1d(label,fff_array_get1d(neighb,j));
				
		if (k>-1){
		  while (fff_array_get1d(father,k)!=k) 
			k = fff_array_get1d(father,k);
		  for (l=0 ; l<q ; l++)
			if (fff_array_get1d(possible,l)>-1)
			  if (fff_array_get1d(possible,l)==k)
				break;
		  if (fff_array_get1d(possible,l)!=k){
			if (l>1) {
			  /* printf("%ld %ld %ld",i,q, l);
			     for (m=0 ; m<l+1 ; m++) printf(" %ld ",fff_array_get1d(possible,m));
			     printf("\n"); */
			}
			fff_array_set1d(possible,q,k);
			q++;
		  }
		}
	  }
	  
	  if (q==0){
		fff_array_set1d(label,win,ll);
		fff_array_set1d(idx,ll,win);
		fff_vector_set(height,ll,fff_vector_get(field,win));
		ll++;
	  }
	  if (q==1)
		fff_array_set1d(label,win,fff_array_get1d(possible,0));
	  if (q>1){
	    /* birfurcation : create a new label */
		for (j=0 ; j<q ; j++){
		  k =  fff_array_get1d(possible,j);
		  fff_array_set1d(father,k,ll);
		}
		fff_array_set1d(label,win,ll);
		fff_array_set1d(idx,ll,win);
		fff_vector_set(height,ll,fff_vector_get(field,win));
		ll++;
	  }
	}
  }
  
    
  papa = fff_array_new1d(FFF_LONG,ll);
  indices = fff_array_new1d(FFF_LONG,ll);
  hauteur = fff_vector_new(ll);
  for (i=0 ; i<ll ; i++){
	fff_array_set1d(papa,i,fff_array_get1d(father,i));
	fff_array_set1d(indices,i,fff_array_get1d(idx,i));
	fff_vector_set(hauteur,i,fff_vector_get(height,i));
  }
  *Father = papa;
  *Height = hauteur;
  *Idx = indices;

  fff_array_delete(cindices);
  fff_array_delete(neighb);
  fff_vector_delete(weight);
  
  fff_array_delete(possible);
  fff_array_delete(father);
  fff_array_delete(idx);
  fff_vector_delete(height);
  free(p);

  return(ll);
}
static long  _fff_list_add( long *listn, double *listd,  const long newn, const double newd, const long k, const long j)
{  
  long i = k; 
  while (listd[i-1] > newd){
	if (i==j)break;
    listd[i] = listd[i-1];
    listn[i] = listn[i-1];
    i--;
  } 
  i = FFF_MAX(i,j);
  listd[i] = newd;
  listn[i] = newn;
  return(0);
}

static long _fff_list_move( long *listn, double *listd,  const long newn, const double newd, const long k, const long j)
{ 
  char* proc = "_fff_list_move"; 
  long i = k-1;
  while (listn[i]!=newn) {
    i--;
    if (i<j){
	  long m;
	  for (m=0 ; m<k ; m++) if (listn[m]==newn) printf("found %ld %ld \n",m,listn[m]);
	  printf("\n");
	  printf("%s %ld %ld %ld \n",proc,newn,k,j);
    }
  }
  if (i>=j){
	while (listd[i-1]>newd){
	  if (i==j)break;
	  listd[i] = listd[i-1];
	  listn[i] = listn[i-1];
	  i--;
	}
	i = FFF_MAX(i,j);
	listd[i] = newd;
	listn[i] = newn;
  }
  return(0);
}

extern long fff_field_voronoi(fff_array *label, const fff_graph* G,const fff_matrix* field,const  fff_array *seeds)
{
  long i,j,k,l,win,start, end;
  long sp = seeds->dimX;
  double infdist = 1.0;
  long V = G->V;
  long E = G->E;
  long ri = 0;
  double w;
  double dsmin,dsmax;
  long smin, smax, lwin; 

  fff_vector *dist, *dg, *weight; 
  fff_array *lg, *cindices, *neighb;
 
  fff_array * visited;
  fff_matrix * feature;
  fff_vector * x;
  fff_vector * y;

  /* argument checking */
  if ((label->dimX)!=V){
    FFF_ERROR("incompatible matrix size \n",EDOM);
  }
  
  infdist = FFF_POSINF;
  
  fff_array_extrema ( &dsmin, &dsmax, seeds );
  smin = (long) dsmin;
  smax = (long) dsmax;
 
  if ((smin<0)|(smax>V-1)){
    FFF_ERROR("seeds have incorrect indices \n",EDOM);
  }

  /* initializations*/
  dist = fff_vector_new(V);
  dg = fff_vector_new(V+1);
  lg = fff_array_new1d(FFF_LONG,V+1);
  cindices = fff_array_new1d(FFF_LONG,V+1);
  neighb = fff_array_new1d(FFF_LONG,E);
  weight = fff_vector_new(E);
  visited = fff_array_new1d(FFF_LONG,V);
  fff_array_set_all(visited,0);
  ri = fff_graph_to_neighb(cindices, neighb, weight,G);
  
  /* create a feature matrix*/
  feature = fff_matrix_new(seeds->dimX,field->size2);
  x = fff_vector_new(field->size2);
  y = fff_vector_new(field->size2);

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
	  fff_matrix_get_row(x,field,win);
	  fff_matrix_set_row(feature,i,x);
	  k++;
	}
    fff_vector_set(dist,win,0);
    fff_vector_set(dg,i,0);
  } 
  win = fff_array_get1d(seeds,0);

  /* iterations */
  for (j=1 ; j<V ; j++){
	fff_array_set1d(visited,win,1);
    start = fff_array_get1d(cindices,win);
    end = fff_array_get1d(cindices,win+1);
    
    for (i=start ; i<end ; i++){
	  /* compute the distance*/
      l = fff_array_get1d(neighb,i);
	  lwin  = fff_array_get1d(label, win);
	  if (fff_array_get1d(visited,l)==0){
		fff_matrix_get_row(x,feature,lwin);
		fff_matrix_get_row(y,field,l);
		fff_vector_sub(x,y);
		w = fff_blas_ddot(x,x);
		
		if ( w < fff_vector_get(dist,l)){	  
		  if (fff_vector_get(dist,l) < infdist)
			ri += _fff_list_move(lg->data, dg->data, l, w, k,j);
		  else{
			ri += _fff_list_add(lg->data, dg->data, l, w, k,j);
			k++; 
		  }
		  fff_vector_set(dist,l,w);
		  fff_array_set1d(label,l, lwin);
		}
	  }
    }
    win = fff_array_get1d(lg,j);
    if (win == -1) break;
    
  }
  fff_array_delete(visited);
  fff_vector_delete(x);
  fff_vector_delete(y);
  fff_matrix_delete(feature);
  fff_array_delete(cindices);
  fff_array_delete(neighb);
  fff_vector_delete(dg);
  fff_vector_delete(dist);
  fff_array_delete(lg);
  fff_vector_delete(weight);

  return(ri);
}
