#include "fffpy.h"
#include <fff_field.h>
#include "fff_array.h"

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API



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
