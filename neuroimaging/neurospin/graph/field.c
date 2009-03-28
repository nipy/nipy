#include "fffpy.h"
#include <fff_field.h>
#include "fff_array.h"

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API

/* Code pour la creation du module */ 
static char diffusion_doc[] = 
" field = diffusion(a,b,d,field,nbiter=1)\n\
  diffusion of a field of data in a weighted graph structure\n\
 INPUT :\n\
 - (a,b,d) sparse coding of the adjacency matrix of the graph \n\
    See _graph for more detail \n\
 - field is an (n,p) array of data that has to be smoothed \n\
  n = number of vertices of the graph \n\
  p = dimension of the firld \n\
 - nbiter : the number of iterations required \n\
  (the larger the smoother) \n\
 OUTPUT:\n\
 - field:   the resulting smoothed field\n\
 ";

static char dilation_doc[] = 
" field = dilation(a,b,field,nbiter=1)\n\
  Morphological dilation of a field of values in a graph structure\n\
 INPUT :\n\
 - (a,b) sparse coding of the adjacency matrix of the graph \n\
    See _graph for more detail \n\
 - field is an (n) array of data that has to be dilated \n\
  n = number of vertices of the graph \n\
 - nbiter : the number of iterations required \n\
  (the larger the smoother) \n\
 OUTPUT:\n\
 - field:   the resulting dilated field\n\
 ";

static char erosion_doc[] = 
" field = erosion(a,b,field,nbiter=1)\n\
  Morphological erosion of a field of values in a graph structure\n\
 INPUT :\n\
 - (a,b) sparse coding of the adjacency matrix of the graph \n\
    See _graph for more detail \n\
 - field is an (n) array of data that has to be eroded \n\
  n = number of vertices of the graph \n\
 - nbiter : the number of iterations required \n\
  (the larger the smoother) \n\
 OUTPUT:\n\
 - field:   the resulting eroded field\n\
 ";

static char closing_doc[] = 
" field = closing(a,b,field,nbiter=1)\n\
  Morphological closing of a field of values in a graph structure\n\
 INPUT :\n\
 - (a,b) sparse coding of the adjacency matrix of the graph \n\
    See _graph for more detail \n\
 - field is an (n) array of data that has to be closed \n\
  n = number of vertices of the graph \n\
 - nbiter : the number of iterations required \n\
   OUTPUT:\n\
 - field:   the resulting closed field\n\
 ";

static char opening_doc[] = 
" field = opening(a,b,field,nbiter=1)\n\
  Morphological opening of a field of values in a graph structure\n\
 INPUT :\n\
 - (a,b) sparse coding of the adjacency matrix of the graph \n\
    See _graph for more detail \n\
 - field is an (n) array of data that has to be opened \n\
  n = number of vertices of the graph \n\
 - nbiter : the number of iterations required \n\
   OUTPUT:\n\
 - field:   the resulting opened field\n\
 ";

static char local_maxima_doc[] = 
" depth = local_maxima(a,b,field)\n\
  Look for the local maxima of a field over a certain graph structure\n\
 INPUT :\n\
 - (a,b) sparse coding of the adjacency matrix of the graph \n\
    See _graph for more detail \n\
 - field is an (n) array of data that has to be smoothed \n\
  n = number of vertices of the graph \n\
 OUTPUT:\n\
 - depth: a labelling of the vertices such that \n\
depth[v] = 0 if v is not a local maximum  \n\
depth[v] = 1 if v is a first order maximum \n\
...\n\
depth[v] = q if v is a q-order maximum \n\
 ";


static char get_local_maxima_doc[] = 
" idx,depth = get_local_maxima(a,b,field,th=-infty)\n\
  Look for the local maxima of a field over a certain graph structure\n\
 INPUT :\n\
 - (a,b) sparse coding of the adjacency matrix of the graph \n\
    See _graph for more detail \n\
 - field is an (n) array of data that has to be smoothed \n\
  n = number of vertices of the graph \n\
 - th is a threshold so that only values above th are considered \n\
   by default, th = -infty (numpy)\n\
 OUTPUT:\n\
 - idx: the indices of the vertices that are local maxima \n\
 - depth: the depth of the local maxima : \n\
depth[idx[i]] = q means that idx[i] is a q-order maximum \n\
 ";

static char custom_watershed_doc[] = 
" idx,depth, major,label = custom_watershed(a,b,field,th=-infty)\n\
  perfoms a watershed analysis of the field.\n\
Note that bassins are found aound each maximum \n\
(and not minimum as conventionally) \n\
 INPUT :\n\
 - (a,b) sparse coding of the adjacency matrix of the graph \n\
    See _graph for more detail \n\
 - field is an (n) array of data that has to be smoothed \n\
  n = number of vertices of the graph \n\
 - th is a threshold so that only values above th are considered \n\
   by default, th = -infty (numpy)\n\
 OUTPUT:\n\
 - idx: the indices of the vertices that are local maxima \n\
 - depth: the depth of the local maxima \n\
depth[idx[i]] = q means that idx[i] is a q-order maximum \n\
Note that this is also the diameter of the basins \n\
associated with local maxima \n\
- major: the label of the maximum which dominates each local maximum \n\
i.e. it describes the hierarchy of the local maxima \n\
- label : a labelling of thevertices according to their bassin \n\
idx, depth and major have length q, where q is the number of bassins \n\
label as length n: the number of vertices \n\
 ";


static char threshold_bifurcations_doc[] = 
" idx,height,father,labels = threshold_bifurcations(a,b,field,th=-infty)\n\
  perfoms a bifurcation analysis of the field.\n\
   Bifurcations are defined as changes in the topology in the level sets \n\
   when the level (threshold) is varied \n\
   This can been thought of as a kind of Morse description \n\
 INPUT :\n\
 - (a,b) sparse coding of the adjacency matrix of the graph \n\
    See _graph for more detail \n\
 - field is an (n) array of data that has to be smoothed \n\
  n = number of vertices of the graph \n\
 - th is a threshold so that only values above th are considered \n\
   by default, th = -infty (numpy)\n\
 OUTPUT:\n\
 - idx: the indices of the vertices that are maxima or saddle points \n\
   associated uniquely with the level sets \n\
 - height : this is the maximal height of the field for a given label \n\
 - father: is the father (in the tree sense of the inclusion relation) \n \
   of each set \n\
 - labels is the labelling of the input data according to the set \n\
   it belongs to. its value is -1 for parts of the fiels below th \n\
   label as length n: the number of vertices \n\
 ";


static char field_voronoi_doc[] = 
" labels = threshold_bifurcations(a,b,field,seed)\n \
  performs a nearest-neighbour labelling of the field starting from the seeds \n\
   INPUT:\n\
    - (a,b) sparse coding of the adjacency matrix of the graph \n\
    See _graph for more detail \n\
 - field is an (n) array of data that has to be smoothed \n\
  n = number of vertices of the graph \n\
 -seed is an array yielding the vertex numbers of the seeds\n\
   OUTPUT:\n\
  - labels is the labelling of the input data according to the set \n\
";


static char module_doc[] = 
" Field processing (smoothing, morphology) routines.\n\
Here, a field is defined as arrays [eA,eB,eD,vF] where.\n\
(eA,eB,eC) define a graph and \n\
vF a dunction defined on the edges of the graph \n\
Author: Bertrand Thirion (INRIA Futurs, Orsay, France), 2004-2006.";

static PyArrayObject* diffusion(PyObject* self, PyObject* args)
{
   PyArrayObject *a, *b, *d, *f;
   int V,E,i,iter=1;
  
  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!O!|i:diffusion", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &d,
			     &PyArray_Type, &f,
			     &iter
			     ); 
  if (!OK) return NULL;   

  /* prepare C arguments */
  fff_array* A = fff_array_fromPyArray( a ); 
  fff_array* B = fff_array_fromPyArray( b );
  fff_vector* D = fff_vector_fromPyArray(d);
  E = A->dimX;
  
  /* Make a copy of the field matrix to avoid it being modified */
  /* this is rather nasty, sorry */
  fff_matrix *ftemp = fff_matrix_fromPyArray(f);
  fff_matrix *field = fff_matrix_new(ftemp->size1,ftemp->size2);
  fff_matrix_memcpy (field, ftemp);
  fff_matrix_delete(ftemp);
  V = field->size1;
    
  /* do the job */
  
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  if (G==NULL)
    return NULL;
  fff_array_delete(A);
  fff_array_delete(B);
  fff_vector_delete(D);
  
  for (i=0 ; i<iter ; i++)
    fff_field_md_diffusion(field, G);
  
  
  fff_graph_delete(G);
  
  /* get the results as python arrrays*/  
  f = fff_matrix_toPyArray( field ); 
  
  return f;
}

static PyArrayObject* dilation(PyObject* self, PyObject* args)
{
   PyArrayObject *a, *b, *f;
   int V,E,iter=1;
  
  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!|i:dilation", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &f,
			     &iter
			     ); 
  if (!OK) return NULL;   

  /* prepare C arguments */
  fff_array* A = fff_array_fromPyArray( a ); 
  fff_array* B = fff_array_fromPyArray( b );  
  E = A->dimX;
  fff_vector* D = fff_vector_new(E);

  fff_vector *field = fff_vector_fromPyArray(f);
  V = field->size;
    
  /* do the job */
  
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  if (G==NULL)
    return NULL;
  fff_array_delete(A);
  fff_array_delete(B);
  fff_vector_delete(D);
  
  fff_field_dilation(field, G,iter);
    
  fff_graph_delete(G);
  
  /* get the results as python arrrays*/
  f = fff_vector_toPyArray( field ); 
  
  return f;
}

static PyArrayObject* erosion(PyObject* self, PyObject* args)
{
   PyArrayObject *a, *b, *f;
   int V,E,iter=1;
  
  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!|i:erosion", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &f,
			     &iter
			     ); 
  if (!OK) return NULL;   

  /* prepare C arguments */
  fff_array* A = fff_array_fromPyArray( a ); 
  fff_array* B = fff_array_fromPyArray( b );  
  E = A->dimX;
  fff_vector* D = fff_vector_new(E);

  fff_vector *field = fff_vector_fromPyArray(f);
  V = field->size;
    
  /* do the job */
  
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  if (G==NULL)
    return NULL;
  fff_array_delete(A);
  fff_array_delete(B);
  fff_vector_delete(D);
  
  fff_field_erosion(field, G,iter);
    
  fff_graph_delete(G);
  
  /* get the results as python arrrays*/
  f = fff_vector_toPyArray( field ); 
  
  return f;
}

static PyArrayObject* opening(PyObject* self, PyObject* args)
{
   PyArrayObject *a, *b, *f;
   int V,E,iter=1;
  
  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!|i:opening", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &f,
			     &iter
			     ); 
  if (!OK) return NULL;   

  /* prepare C arguments */
  fff_array* A = fff_array_fromPyArray( a ); 
  fff_array* B = fff_array_fromPyArray( b );  
  E = A->dimX;
  fff_vector* D = fff_vector_new(E);

  fff_vector *field = fff_vector_fromPyArray(f);
  V = field->size;
    
  /* do the job */
  
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  if (G==NULL)
    return NULL;
  fff_array_delete(A);
  fff_array_delete(B);
  fff_vector_delete(D);
  
  fff_field_opening(field, G,iter);
    
  fff_graph_delete(G);
  
  /* get the results as python arrrays*/
  f = fff_vector_toPyArray( field ); 
  
  return f;
}


static PyArrayObject* closing(PyObject* self, PyObject* args)
{
   PyArrayObject *a, *b, *f;
   int V,E,iter=1;
  
  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!|i:closing", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &f,
			     &iter
			     ); 
  if (!OK) return NULL;   

  /* prepare C arguments */
  fff_array* A = fff_array_fromPyArray( a ); 
  fff_array* B = fff_array_fromPyArray( b );  
  E = A->dimX;
  fff_vector* D = fff_vector_new(E);

  fff_vector *field = fff_vector_fromPyArray(f);
  V = field->size;
    
  /* do the job */
  
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  if (G==NULL)
    return NULL;
  fff_array_delete(A);
  fff_array_delete(B);
  fff_vector_delete(D);
  
  fff_field_closing(field, G,iter);
    
  fff_graph_delete(G);
  
  /* get the results as python arrrays*/
  f = fff_vector_toPyArray( field ); 
  
  return f;
}

static PyArrayObject* local_maxima(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *de, *f;
  int V,E;
  
  // Parse input 
  // see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html
  int OK = PyArg_ParseTuple( args, "O!O!O!:local_maxima", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &f
			     ); 
  if (!OK) return NULL;   
 

  // prepare C arguments
  fff_array* A = fff_array_fromPyArray( a ); 
  fff_array* B = fff_array_fromPyArray( b );

  E = A->dimX;
  
  fff_vector *field = fff_vector_fromPyArray(f);
  V = field->size;
  fff_vector* D = fff_vector_new(E);
    
  // do the job 
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  if (G==NULL)
    return NULL;
  fff_array_delete(A);
  fff_array_delete(B);
  fff_vector_delete(D);
  
  fff_array* depth = fff_array_new1d(FFF_LONG, V );
  
  fff_field_maxima(depth, G, field);
  
  fff_graph_delete(G);
  fff_vector_delete(field);
  // get the results as python arrrays
  
  de = fff_array_toPyArray( depth ); 

  return de;

  
}

static PyObject* get_local_maxima(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *dep,*idx, *f;
   int V,E;
  double th = FFF_NEGINF;
  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!|d:get_local_maxima", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &f,
			     &th
			     ); 
  if (!OK) return NULL;   

  /* prepare C arguments */
  fff_array* A = fff_array_fromPyArray( a ); 
  fff_array* B = fff_array_fromPyArray( b );
  
  E = A->dimX;
  
  fff_vector *field = fff_vector_fromPyArray(f);
  V = field->size;
  fff_vector* D = fff_vector_new(E);
    
  /* do the job */
  
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  if (G==NULL)
    return NULL;
  fff_array_delete(A);
  fff_array_delete(B);
  fff_vector_delete(D);
  
  fff_array* depth;
  fff_array* indices; 
  OK = fff_field_get_maxima_th(&depth, &indices, G, field,th);
  
  fff_graph_delete(G);
  fff_vector_delete(field);
  
  /* get the results as python arrrays*/  
  PyObject* ret = NULL;
  if (OK>0){
	dep = fff_array_toPyArray( depth ); 
	idx = fff_array_toPyArray( indices);
	ret = Py_BuildValue("NN",idx,dep); 
  }
  else{
	dep = NULL;
	idx = NULL;
  }
	  
  /* Output tuple */

  return ret;
}

static PyObject* custom_watershed(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *dep,*idx,*maj,*lab, *f;
  int V,E,k;
  double th =  FFF_NEGINF;
  
  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!|d:custom_watershed", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &f,
			     &th
			     ); 
  if (!OK) return NULL;   

  /* prepare C arguments */
  fff_array* A = fff_array_fromPyArray( a ); 
  fff_array* B = fff_array_fromPyArray( b );
  
  E = A->dimX;
  
  fff_vector *field = fff_vector_fromPyArray(f);
  V = field->size;
  /* printf("%d %d \n",E,V);*/
  fff_vector* D = fff_vector_new(E);
    
  /* do the job */
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  if (G==NULL) 
    return NULL;
  fff_array_delete(A);
  fff_array_delete(B);
  fff_vector_delete(D);
  
  fff_array* depth;
  fff_array* indices; 
  fff_array* major;
  fff_array *label = fff_array_new1d(FFF_LONG,V);

  if (th == FFF_NEGINF)
    k = fff_custom_watershed( &indices, &depth, &major, label, field, G);
  else
    k = fff_custom_watershed_th( &indices, &depth, &major, label, field, G,th);
  /*  printf("%d \n",k);*/
  fff_graph_delete(G);
  fff_vector_delete(field);

  /* get the results as python arrrays*/  
  lab = fff_array_toPyArray( label);
  
  if (k>0){
    dep = fff_array_toPyArray( depth ); 
    idx = fff_array_toPyArray( indices);
    maj = fff_array_toPyArray( major);  
  }
  else{
    dep = fffpyZeroLONG(); 
    idx = fffpyZeroLONG(); 
    maj = fffpyZeroLONG(); 
  }

  PyObject* ret = Py_BuildValue("NNNN",idx,dep,maj,lab); 

  return ret;
}

static PyObject* threshold_bifurcations(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *hei,*idx,*father,*lab, *f;
  int V,E,k;
  double th =  FFF_NEGINF;
  
  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!|d:threshold_bifurcations", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &f,
			     &th
			     ); 
  if (!OK) return NULL;   

  /* prepare C arguments */
  fff_array* A = fff_array_fromPyArray( a ); 
  fff_array* B = fff_array_fromPyArray( b );
  
  E = A->dimX;
  
  fff_vector *field = fff_vector_fromPyArray(f);
  V = field->size;
  /* printf("%d %d \n",E,V);*/
  fff_vector* D = fff_vector_new(E);
    
  /* do the job */
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  if (G==NULL) 
    return NULL;
  fff_array_delete(A);
  fff_array_delete(B);
  fff_vector_delete(D);
  
  fff_vector* height;
  fff_array* indices; 
  fff_array* major;
  fff_array *label = fff_array_new1d(FFF_LONG,V);

  
  k = fff_field_bifurcations( &indices, &height, &major,label,field,G, th);
  fff_graph_delete(G);
  fff_vector_delete(field);

  /* get the results as python arrrays*/  
  lab = fff_array_toPyArray( label);

  if (k>0){
    hei = fff_vector_toPyArray( height ); 
    idx = fff_array_toPyArray( indices);
    father = fff_array_toPyArray( major);  
  }
  else{
    hei = 0; 
    idx = fffpyZeroLONG(); 
    father = fffpyZeroLONG(); 
  }

  PyObject* ret = Py_BuildValue("NNNN",idx,hei,father,lab); 

  return ret;
}

static PyObject* field_voronoi(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *f, *seed, *label;
  int V,E;
  
  
  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!O!|d:threshold_bifurcations", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &f,
				 &PyArray_Type, &seed
			     ); 
  if (!OK) return NULL;   

  /* prepare C arguments */
  fff_array* A = fff_array_fromPyArray( a ); 
  fff_array* B = fff_array_fromPyArray( b );
  
  E = A->dimX;
  fff_vector* D = fff_vector_new(E);
  fff_matrix *Field = fff_matrix_fromPyArray(f);
  V = Field->size1;
  
  /* do the job */
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  if (G==NULL) 
    return NULL;
  fff_array_delete(A);
  fff_array_delete(B);
  fff_vector_delete(D);
  
  fff_array* Seed = fff_array_fromPyArray( seed ); 
 
  fff_array *Label = fff_array_new1d(FFF_LONG,V);
  fff_field_voronoi(Label, G,Field, Seed);

  fff_graph_delete(G);
  fff_matrix_delete(Field);
  fff_array_delete(Seed);

  /* get the results as python arrrays*/  
  label = fff_array_toPyArray( Label);

  PyObject* ret = Py_BuildValue("N",label); 

  return ret;
}

static PyMethodDef module_methods[] = {
  {"diffusion",    /* name of func when called from Python */
   (PyCFunction)diffusion,      /* corresponding C function */
   METH_KEYWORDS,   /* ordinary (not keyword) arguments */
   diffusion_doc}, /* doc string */
  {"dilation",    /* name of func when called from Python */
   (PyCFunction)dilation,      /* corresponding C function */
   METH_KEYWORDS,   /* ordinary (not keyword) arguments */
   dilation_doc}, /* doc string */
    {"erosion",    /* name of func when called from Python */
   (PyCFunction)erosion,      /* corresponding C function */
   METH_KEYWORDS,   /* ordinary (not keyword) arguments */
   erosion_doc}, /* doc string */
    {"opening",    /* name of func when called from Python */
   (PyCFunction)opening,      /* corresponding C function */
   METH_KEYWORDS,   /* ordinary (not keyword) arguments */
   opening_doc}, /* doc string */
    {"closing",    /* name of func when called from Python */
   (PyCFunction)closing,      /* corresponding C function */
   METH_KEYWORDS,   /* ordinary (not keyword) arguments */
   closing_doc}, /* doc string */
  {"local_maxima",    /* name of func when called from Python */
   (PyCFunction)local_maxima,      /* corresponding C function */
   METH_KEYWORDS,   /* ordinary (not keyword) arguments */
   local_maxima_doc}, /* doc string */
   {"get_local_maxima",    /* name of func when called from Python */
   (PyCFunction)get_local_maxima,      /* corresponding C function */
   METH_KEYWORDS,   /* ordinary (not keyword) arguments */
   get_local_maxima_doc}, /* doc string */
  {"threshold_bifurcations",    /* name of func when called from Python */
   (PyCFunction)threshold_bifurcations,      /* corresponding C function */
   METH_KEYWORDS,   /* ordinary (not keyword) arguments */
   threshold_bifurcations_doc}, /* doc string */
  {"custom_watershed",    /* name of func when called from Python */
   (PyCFunction)custom_watershed,      /* corresponding C function */
   METH_KEYWORDS,   /* ordinary (not keyword) arguments */
   custom_watershed_doc}, /* doc string */
  {"field_voronoi",    /* name of func when called from Python */
   (PyCFunction)field_voronoi,      /* corresponding C function */
   METH_KEYWORDS,   /* ordinary (not keyword) arguments */
   field_voronoi_doc}, /* doc string */
  {NULL, NULL,0,NULL}
};


void init_field(void)
{
  Py_InitModule3("_field", module_methods, module_doc);
  fffpy_import_array();
  import_array();   /* required NumPy initialization */
}
