#include "fffpy.h"
#include <fff_graphlib.h>
#include "fff_array.h"

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API


/* doc */ 


static char graph_dijkstra_doc[] = 
" dg = graph_dijkstra(a,b,d,seed,V)\n\
  returns all the geodesic distances starting from seed\n\
  d>=0 is mandatory c\n\
 INPUT:\n\
- The edges of the input graph are defined through the couple of 1-d arrays \n\
A,B such that [A[e] B[e]] are the vertices and D[e] an associated attribute \n\
(distance/weight/affinity) \n\
- seed is the edge from which the distances are computed \n\
- V is the numner of vertices of the graph\n\
It is an optional argument, by default v = max(max(a),max(b))+1\n\
OUTPUT:\n\
- the graph distance dg from the seed to any edge \n\
  ";

static char graph_dijkstra_multiseed_doc[] = 
" dg = graph_dijkstra_multiseed(a,b,d,seed,V)\n\
  returns all the geodesic distances starting from seed\n\
  d>=0 is mandatory c\n\
 INPUT:\n\
- The edges of the input graph are defined through the couple of 1-d arrays \n\
A,B such that [A[e] B[e]] are the vertices and D[e] an associated attribute \n\
(distance/weight/affinity) \n\
- seed is the vector of edges from which the distances are computed \n\
- V is the numner of vertices of the graph\n\
It is an optional argument, by default v = max(max(a),max(b))+1\n\
OUTPUT:\n\
- the graph distance dg from the seed to any edge \n\
  ";

static char graph_floyd_doc[] = 
" dg = graph_floyd(a,b,d,seed=NULL,V)\n\
  returns all the geodesic distances starting from seeds\n\
  d>=0 is mandatory and checked in the function c\n\
INPUT:\n\
- The edges of the input graph are defined through the couple of 1-d arrays \n\
A,B such that [A[e] B[e]] are the vertices and D[e] an associated attribute \n\
(distance/weight/affinity) \n\
- seed is an arry of  edges from which the distances are computed \n\
if seed==NULL, then every edge is a seed point\n\
- V is the numner of vertices of the graph\n\
It is an optional argument, by default v = max(max(a),max(b))+1\n\
OUTPUT:\n\
- the graph distance dg from each seed to any edge \n\
Note that it has size (nbseed,nbedges)\n\
  ";

static char graph_voronoi_doc[] = 
" label = graph_voronoi(a,b,d,seed,V)\n\
  performs a voronoi labelling of the graph \n\
  d>=0 is mandatory ; it is checked in the function c\n \
INPUT:\n\
- The edges of the input graph are defined through the couple of 1-d arrays \n\
A,B such that [A[e] B[e]] are the vertices and D[e] an associated attribute \n\
(distance/weight/affinity) \n\
- seed is an arry of  edges from which the distances are computed \n\
- V is the numner of vertices of the graph\n\
It is an optional argument, by default v = max(max(a),max(b))+1\n\
OUTPUT:\n\
- the voronoi labelling of the vertices \n\
Note that it has the size (nbseed)\n\
  ";

static char graph_rd_doc[] = 
" cliques = graph_voronoi(a,b,d,V)\n\
  performs a clique extraction of the graph using replicator dynamics  \n\
  d>=0 is mandatory ; it is checked in the function c\n \
INPUT:\n\
- The edges of the input graph are defined through the couple of 1-d arrays \n\
A,B such that [A[e] B[e]] are the vertices and D[e] an associated attribute \n\
(distance/weight/affinity) \n\
- V is the numner of vertices of the graph\n\
It is an optional argument, by default v = max(max(a),max(b))+1\n\
OUTPUT:\n\
- the labelling of the vertices according to the clique they belong to \n\
Note that it has the size (V)\n\
  ";

static char module_doc[] = 
" Graph routines.\n\
Author: Bertrand Thirion (INRIA Futurs, Orsay, France), 2004-2008.";

static double _fff_array_max1d(const fff_array *farray);
static double _fff_array_max1d(const fff_array *farray)
{
  // returns the index of the max value on a supposedly 1D array
  // quick and dirty implementation
  long i,n = farray->dimX;
  double val,max = (double) fff_array_get1d(farray,0);
  
  for (i=0 ; i<n ; i++){
	val = (double) fff_array_get1d(farray,i);
	if (val>max)
	  max = val;
  }
  return max;
}



/****************************************************************
 ************ Part 2 : graph analysis ***************************
 ***************************************************************/

static PyArrayObject* graph_dijkstra(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *d, *m;
  int seed;
  int eA, eB, V = 0;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!i|i:graph_dijkstra", 

			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &d,
			     &seed,
			     &V
			     ); 
  if (!OK) return NULL; 
    
  /* prepare C arguments */
  fff_array* A = fff_array_fromPyArray( a ); 
  fff_array* B = fff_array_fromPyArray( b );
  fff_vector* D = fff_vector_fromPyArray(d);
  int E = A->dimX;
  if (V<1){
     eA = (int)_fff_array_max1d(A)+1;
     eB = (int)_fff_array_max1d(B)+1;
     if (eA>V) V = eA;
     if (eB>V) V = eB;
  }

  fff_vector *gd = fff_vector_new(V);
  
  /* do the job */
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  fff_array_delete(A);
  fff_array_delete(B);
  fff_vector_delete(D);
  

  fff_graph_dijkstra(gd->data, G, seed);
  fff_graph_delete(G);
  
  /* get the results as python arrrays*/
  m = fff_vector_toPyArray( gd);
  
  return m;
}

static PyArrayObject* graph_dijkstra_multiseed(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *d, *m, *seed;
  int eA, eB, V = 0;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!O!|i:graph_dijkstra_multiseed", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &d,
			     &PyArray_Type, &seed,
			     &V
			     ); 
  if (!OK) return NULL; 
    
  /* prepare C arguments */
  fff_array* A = fff_array_fromPyArray( a ); 
  fff_array* B = fff_array_fromPyArray( b );
  fff_vector* D = fff_vector_fromPyArray(d);
  fff_array* Seed = fff_array_fromPyArray( seed );
  int E = A->dimX;
  if (V<1){
     eA = (int)_fff_array_max1d(A)+1;
     eB = (int)_fff_array_max1d(B)+1;
     if (eA>V) V = eA;
     if (eB>V) V = eB;
  }

  fff_vector *gd = fff_vector_new(V);
  
  /* do the job */
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  fff_array_delete(A);
  fff_array_delete(B);
  fff_vector_delete(D);
  
  fff_graph_Dijkstra_multiseed(gd, G, Seed);
  fff_graph_delete(G);
  fff_array_delete(Seed);
  
  /* get the results as python arrrays*/
  m = fff_vector_toPyArray( gd);
  
  return m;
}

static PyArrayObject* graph_floyd(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *d, *m, *seed;
  int eA, eB, V = 0;
  int ns=0;
  seed = NULL;
  fff_matrix *gd ;
  fff_array* seeds = NULL;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!|O!i:graph_floyd", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &d,
			     &PyArray_Type, &seed,
			     &V
			     ); 
  if (!OK) return NULL;   

  /* prepare C arguments */
  fff_array* A = fff_array_fromPyArray( a ); 
  fff_array* B = fff_array_fromPyArray( b );
  fff_vector* D = fff_vector_fromPyArray(d);
  int E = A->dimX;
  if (V<1){
     eA = (int)_fff_array_max1d(A)+1;
     eB = (int)_fff_array_max1d(B)+1;
     if (eA>V) V = eA;
     if (eB>V) V = eB;
   }
  /* do the job */
  
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  fff_array_delete(A);
  fff_array_delete(B);
  fff_vector_delete(D);
  
  if (seed==NULL){
    gd = fff_matrix_new(V,V); 
    fff_graph_Floyd(gd,G);
  }
  else{ 
    seeds = fff_array_fromPyArray( seed );
    ns = seeds->dimX;
    
    gd = fff_matrix_new(ns,V);
        
    fff_graph_partial_Floyd(gd,G,seeds->data);
    
    fff_array_delete(seeds);
    
    }
  fff_graph_delete(G);
  
  /* get the results as python arrrays*/
  m = fff_matrix_toPyArray( gd);
  
  return m;
}

static PyArrayObject* graph_voronoi(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *d, *l, *seed;
  int eA, eB, V = 0;
  fff_array* seeds;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!O!|i:graph_voronoi", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &d,
			     &PyArray_Type, &seed,
			     &V
			     ); 
  if (!OK) return NULL;   

  /* prepare C arguments */
  fff_array* A = fff_array_fromPyArray( a ); 
  fff_array* B = fff_array_fromPyArray( b );
  fff_vector* D = fff_vector_fromPyArray(d);
  int E = A->dimX;
  if (V<1){
     eA = (int)_fff_array_max1d(A)+1;
     eB = (int)_fff_array_max1d(B)+1;
     if (eA>V) V = eA;
     if (eB>V) V = eB;
   }
  /* do the job */
  
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  fff_array_delete(A);
  fff_array_delete(B);
  fff_vector_delete(D);
  
  fff_array *label = fff_array_new1d(FFF_LONG,V); 
  seeds = fff_array_fromPyArray( seed );
  
  fff_graph_voronoi(label,G,seeds);
  
  fff_array_delete(seeds);
  fff_graph_delete(G);
  
  /* get the results as python arrrays*/
  l = fff_array_toPyArray( label);
  
  return l;
}

static PyArrayObject* graph_rd(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *d, *l;
  int eA, eB, V = 0;
  
  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!|i:graph_rd", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &d,
			     &V
			     ); 
  if (!OK) return NULL;   

  /* prepare C arguments */
  fff_array* A = fff_array_fromPyArray( a ); 
  fff_array* B = fff_array_fromPyArray( b );
  fff_vector* D = fff_vector_fromPyArray(d);
  int E = A->dimX;
  if (V<1){
     eA = (int)_fff_array_max1d(A)+1;
     eB = (int)_fff_array_max1d(B)+1;
     if (eA>V) V = eA;
     if (eB>V) V = eB;
   }
  /* do the job */
  
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  fff_array_delete(A);
  fff_array_delete(B);
  fff_vector_delete(D);
  
  fff_array *label = fff_array_new1d(FFF_LONG,V); 
    
  fff_graph_cliques(label,G);
  
  fff_graph_delete(G);
  /* get the results as python arrrays*/
  l = fff_array_toPyArray( label);
  
  return l;
}


static PyMethodDef module_methods[] = {
  {"graph_dijkstra",        /* name of func when called from Python */
   (PyCFunction)graph_dijkstra,          /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_dijkstra_doc},        /* doc string */
  {"graph_dijkstra_multiseed", /* name of func when called from Python */
   (PyCFunction)graph_dijkstra_multiseed, /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_dijkstra_multiseed_doc},        /* doc string */
  {"graph_floyd",        /* name of func when called from Python */
   (PyCFunction)graph_floyd,          /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_floyd_doc},        /* doc string */
   {"graph_voronoi",        /* name of func when called from Python */
   (PyCFunction)graph_voronoi,          /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_voronoi_doc},        /* doc string */
   {"graph_rd",        /* name of func when called from Python */
   (PyCFunction)graph_rd,          /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_rd_doc},        /* doc string */
    {NULL, NULL,0,NULL}

};


void init_graph(void)
{
  Py_InitModule3("_graph", module_methods, module_doc);
  fffpy_import_array();
  import_array();   /* required NumPy initialization */
}
