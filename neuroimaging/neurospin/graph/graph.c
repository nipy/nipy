#include "fffpy.h"
#include <fff_graphlib.h>
#include "fff_array.h"
#include <fff_BPmatch.h>

#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API


/* doc */ 
static char graph_complete_doc[] = 
" (A,B,D) = graph_complete(v)\n\
Returns the list of edges of a complete graph with v vertices \n\
i.e. it is the only graph with v*v edges\n\
note that trivial edges are included \n\
INPUT : \n\
the number of vertices of the graph\n\
OUTPUT : \n\
the list of edges [A[e] B[e]] \n\
and a weight/length attribute D[e] \n\
which is initilized with 1 for each non-trivial edge\n\
and 0 for each trivial edge\n\
";

static char graph_knn_doc[] = 
" (A,B,D) = graph_knn(X,k)\n\
  Building the k-nearest-neighbours graph of the data \n\
INPUT:\n\
The array X is assumed to be a n*p feature matrix \n\
where n is the number of features \n\
and p is the dimension of the features \n\
It is assumed that the features are embedded in a (locally) Euclidian space \n\
k is the number of neighbours considered\n\
OUTPUT:\n\
The edges of the resulting (directed) graph are defined through the triplet of 1-d arrays \n\
A,B,D such that [A[e] B[e]] are the vertices D[e] = ||A[e]-B[e]|| Euclidian.\n\
NB:\n\
- the knn system is symmeterized: if (A[e] B[e] D[e]) is one of the edges \n\
then (B[e] A[e] D[e]) is another edge \n\
- trivial neighbours (v v 0) are not included.\n\
- for the sake of speed it is advisable to give a PCA-preprocessed matrix X.\n\
  ";

static char graph_eps_doc[] = 
" (A,B,D) = graph_eps(X,eps)\n\
Building the epsilon-nearest-neighbours graph of the data\n\
INPUT:\n\
The array X is assumed to be a n*p feature matrix \n\
where n is the number of features \n\
and p is the dimension of the features \n\
It is assumed that the features are embedded in a (locally) Euclidian space \n\
epsilon is the number of neighbourood size considered \n\
OUTPUT:\n\
The edges of the resulting (directed) graph are defined through the triplet of 1-d arrays \n\
A,B,D such that [A[e] B[e]] are the vertices D[e] = ||A[e]-B[e]|| Euclidian.\n\
NB:\n\
- trivial neighbours (v v 0) are included.\n\
- for the sake of speed it is advisable to give a PCA-preprocessed matrix X.\n\
";

static char graph_cross_knn_doc[] = 
" (A,B,D) = graph_cross_knn(X,Y,k)\n\
  Building the cross-knn graph of the data \n\
INPUT:\n\
The arrays X  and Y is assumed to be a n1*p and n2*p feature matrices \n\
where (n1,n2) are the number of features in either candy \n\
and p is the dimension of the features \n\
It is assumed that the features are embedded in a (locally) Euclidian space \n\
 is the number of neighbours considered \n\
OUTPUT:\n\
The edges of the resulting (directed) graph are defined through the triplet of 1-d arrays \n\
A,B,D such that [A[e] B[e]] are the vertices D[e] = ||A[e]-B[e]|| Euclidian.\n\
NB:\n\
- for the sake of speed it is advisable to give PCA-preprocessed matrices X and Y. \n\
  ";

static char graph_cross_eps_doc[] = 
" (A,B,D) = graph_cross_eps(X,Y,eps)\n\
  Building the cross_eps graph of the data \n\
INPUT:\n\
The arrays X  and Y is assumed to be a n1*p and n2*p feature matrices \n\
where (n1,n2) are the number of features in either candy \n\
and p is the dimension of the features \n\
It is assumed that the features are embedded in a (locally) Euclidian space \n\
epsilon is the number of neighbourood size considered \n\
OUTPUT:\n\
The edges of the resulting (directed) graph are defined through the triplet of 1-d arrays \n\
A,B,D such that [A[e] B[e]] are the vertices D[e] = ||A[e]-B[e]|| Euclidian.\n\
NB:\n\
- for the sake of speed it is advisable to give PCA-preprocessed matrices X and Y. \n\
  ";

static char graph_cross_eps_robust_doc[] = 
" (A,B,D) = graph_cross_eps_robust(X,Y,eps)\n\
  Building the cross_eps graph of the data \n\
INPUT:\n\
The arrays X  and Y is assumed to be a n1*p and n2*p feature matrices \n\
where (n1,n2) are the number of features in either candy \n\
and p is the dimension of the features \n\
It is assumed that the features are embedded in a (locally) Euclidian space \n\
epsilon is the number of neighbourood size considered \n\
OUTPUT:\n\
The edges of the resulting (directed) graph are defined through the triplet of 1-d arrays \n\
A,B,D such that [A[e] B[e]] are the vertices D[e] = ||A[e]-B[e]|| Euclidian.\n\
NB:\n\
- for the sake of speed it is advisable to give PCA-preprocessed matrices X and Y. \n\
  ";

static char graph_mst_doc[] = 
" (A,B,D) = graph_mst(X)\n\
  Building the MST of the data \n\
INPUT:\n\
The array X is assumed to be a n*p feature matrix \n\
where n is the number of features \n\
and p is the dimension of the features \n\
It is assumed that the features are embedded in a (locally) Euclidian space \n\
OUTPUT:\n\
The edges of the resulting (directed) graph are defined through the triplet of 1-d arrays \n\
A,B,D such that [A[e] B[e]] are the vertices D[e] = ||A[e]-B[e]|| Euclidian.\n\
NB:\n\
- The edge system is symmeterized: if (A[e] B[e] D[e]) is one of the edges \n\
then (B[e] A[e] D[e]) is another edge \n\
- As a consequence, the graph comprises (2n-2) edges \n\
  ";

static char graph_skeleton_doc[] = 
" (A,B,D) = graph_mst(A1,B1,D1)\n\
  Building the MST of the data \n\
INPUT:\n\
The array X is assumed to be a n*p feature matrix \n\
where n is the number of features \n\
and p is the dimension of the features \n\
It is assumed that the features are embedded in a (locally) Euclidian space \n\
OUTPUT:\n\
The edges of the resulting (directed) graph are defined through the triplet of 1-d arrays \n\
A,B,D such that [A[e] B[e]] are the vertices D[e] = ||A[e]-B[e]|| Euclidian.\n\
NB:\n\
- The edge system is symmeterized: if (A[e] B[e] D[e]) is one of the edges \n\
then (B[e] A[e] D[e]) is another edge \n\
- As a consequence, the graph comprises (2n-2) edges \n\
  ";

static char graph_3d_grid_doc[] = 
" (A,B,D) = graph_3d_grid(XYZ,k)\n\
  Building the 6-nn, 18-nn or 26nn of the data, \n\
which are sampled on a three-dimensional grid \n\
 INPUT:\n\
- The array XYZ is assumed to be a n*3 coordinate matrix \n\
which are assumed to have integer values.  \n\
- k (=6 or 18 or 26) is the neighboring system considered \n\
 OUTPUT:\n\
The edges of the resulting (directed) graph are defined through the triplet of 1-d arrays \n\
A,B,D such that [A[e] B[e]] are the vertices D[e] = ||A[e]-B[e]|| Euclidian.\n\
NB:\n\
- The edge system is symmeterized: if (A[e] B[e] D[e]) is one of the edges \n\
then (B[e] A[e] D[e]) is another edge \n\
- trivial edges are included, and have distance 0.\n\
  ";

static char graph_degrees_doc[] = 
" (r,l) = graph_degrees(a,b,v)\n\
  Computation of the left- and right-degree of the graph vertices\n\
 INPUT:\n\
- The edges of the input graph are defined through the couple of 1-d arrays \n\
A,B such that [A[e] B[e]] are the vertices \n\
- v is the number of vertices of the graph \n\
It is an optional argument, by default v = max(max(a),max(b))+1\n\
OUTPUT:\n\
- 1-d arrays r,l of size v that give the right and left degrees of the graph \n\
  ";

static char graph_adjacency_doc[] = 
" m = graph_adjacency(a,b,d,V)\n\
Creates the adjancency matrix of the graph \n\
INPUT:\n\
- The edges of the input graph are defined through the couple of 1-d arrays \n\
A,B such that [A[e] B[e]] are the vertices and D[e] an associated attribute \n\
(distance/weight/affinity) \n\
-V is the numner of vertices of the graph\n\
It is an optional argument, by default v = max(max(a),max(b))+1\n\
OUTPUT:\n\
- The array m of size (V,V), with 0 everywhere \n\
but m[A[e],B[e]] = D[e] for each e \n\
  ";

static char graph_reorder_doc[] = 
" a,b,d = graph_reorder(a,b,d,c,V)\n\
Reorder the graph according to the index c\n\
INPUT:\n\
- The edges of the input graph are defined through the couple of 1-d arrays \n\
A,B such that [A[e] B[e]] are the vertices and D[e] an associated attribute \n\
(distance/weight/affinity) \n\
- c is an index that designates the array \n\
according to which the vectors are jointly reordered \n\
It is an optional argument, by default c = 0\n\
c == 0 => reordering makes A[e] increasing, and B[e] increasing for A[e] fixed \n\
c == 1 => reordering makes B[e] increasing, and A[e] increasing for B[e] fixed \n\
c == 2 => reordering makes D[e] increasing \n\
- V is the numner of vertices of the graph\n\
It is an optional argument, by default v = max(max(a),max(b))+1\n\
OUTPUT:\n\
- The arrays A,B,D reordered as required \n\
";

static char graph_normalize_doc[] = 
" a,b,d,s = graph_normalize(a,b,d,c=0,V)\n\
Normalize the graph according to the index c\n\
Normalization means that the sum of the edges values \n\
that go into or out each vertex must sum to 1 \n\
INPUT:\n\
- The edges of the input graph are defined through the couple of 1-d arrays \n\
A,B such that [A[e] B[e]] are the vertices and D[e] an associated attribute \n\
(distance/weight/affinity) \n\
- c is an index that designates the array \n\
according to which D is normalized \n\
It is an optional argument, by default c = 0\n\
c == 0 => for each vertex a, sum{A[e]=a} D[e]=1 \n\
c == 1 => for each vertex b, sum{B[e]=b} D[e]=1 \n\
c == 2 => a symmetric normalization is performed \n\
- V is the numner of vertices of the graph\n\
It is an optional argument, by default v = max(max(a),max(b))+1\n\
OUTPUT:\n\
- The arrays A,B,D normalized as required \n\
- s: the values sum{A[e]=a} D[e](resp sum{B[e]=b} D[e]) \n\
- if c==2, t= sum{B[e]=a} D[e]\n\
note that when sum(A[e]=a) D[e]=0, nothing is performed \n\
";

static char graph_cut_redundancies_doc[] = 
" a,b,d = graph_cut_redundancies(a,b,d,V)\n\
Removes redundant edges from the graph \n\
   i.e. edges e,f,e!=f such that A[e]=A[f] and B[e]=B[f] \n\
   if D[e]!=D[f], only the first one in the list is kept \n\
INPUT:\n\
- The edges of the input graph are defined through the couple of 1-d arrays \n\
A,B such that [A[e] B[e]] are the vertices and D[e] an associated attribute \n\
(distance/weight/affinity) \n\
- V is the numner of vertices of the graph\n\
It is an optional argument, by default v = max(max(a),max(b))+1\n\
OUTPUT:\n\
- The arrays A,B,D with no redundant data  \n\
  ";

static char graph_set_euclidian_doc[] = 
" d = graph_set_euclidian(a,b,X)\n\
Compute the length of the edges of the graph in an Euclidian space  \n\
INPUT:\n\
- The edges of the input graph are defined through the couple of 1-d arrays \n\
A,B such that [A[e] B[e]] are the vertices \n\
- X is the coordinate matrix of the vertices \n\
it is assumed to be of size V*p \n\
where V is the number of vertices \n\
OUTPUT:\n\
- D such that D[e] = ||X[A[e],:]-X[B[e],:]|| \n\
";

static char graph_set_gaussian_doc[] = 
  " d = graph_set_euclidian(a,b,X,sigma = 0)\n\
Compute the value of the edges of the graph \n\
as a gaussian function of their length  \n\
 INPUT:\n\
- The edges of the input graph are defined through the couple of 1-d arrays \n\
A,B such that [A[e] B[e]] are the vertices \n\
- X is the coordinate matrix of the vertices \n\
it is assumed to be of size V*p \n\
where V is the number of vertices \n\
OUTPUT:\n\
 - D such that D[e] = exp(-||X[A[e],:]-X[B[e],:]||^2/(2*sigma^2)) \n \
  Note that when sigma = 0, an empirical value is used : \n\
  sigma = sqrt(mean(||X[A[e],:]-X[B[e],:]||^2)) \n\
  ";

static char graph_to_neighb_doc[] = 
  " ci,ne,we = graph_to_neighb(a,b,d,V)\n\
  converts the graph to a neighboring system\n\
The neighboring system is nothing but a (sparser) representation of the edge matrix\n\
INPUT:\n\
- The edges of the input graph are defined through the couple of 1-d arrays \n\
A,B such that [A[e] B[e]] are the vertices and D[e] an associated attribute \n\
(distance/weight/affinity) \n\
- V is the numner of vertices of the graph\n\
It is an optional argument, by default V = max(max(a),max(b))+1\n\
OUTPUT:\n\
- The arrays ci, ne ,we such that \n\
the set of edge (a[i],b[i],d[i])\n\
is coded such that:\n\
for j in [ci[a] ci[a+1][, (a,eB[j],eD[j]) is an edge of the graph \n\
  ";

static char graph_symmeterize_doc[] = 
  " a,b,d = graph_symmeterize(a,b,d,V)\n\
  symmeterize the graph G(a,b,d), ie produces the graph c\n	\
  whose adjacency matrix would be the symmetric part of \n	\
  the adjacency matrix of G(a,b,d)\n				\
  INPUT:\n\
- The edges of the input graph are defined through the couple of 1-d arrays \n\
A,B such that [A[e] B[e]] are the vertices and D[e] an associated attribute \n\
(distance/weight/affinity) \n\
- V is the numner of vertices of the graph\n\
It is an optional argument, by default V = max(max(a),max(b))+1\n\
OUTPUT:\n\
- A,B,D as required \n\
";

static char graph_antisymmeterize_doc[] = 
  " a,b,d = graph_antisymmeterize(a,b,d,V)\n\
  Antisymmeterize the graph G(a,b,d), ie produces the graph c\n	\
  whose adjacency matrix would be the antisymmetric part of \n	\
  the adjacency matrix of G(a,b,d)\n				\
  INPUT:\n\
- The edges of the input graph are defined through the couple of 1-d arrays \n\
A,B such that [A[e] B[e]] are the vertices and D[e] an associated attribute \n\
(distance/weight/affinity) \n\
- V is the numner of vertices of the graph\n\
It is an optional argument, by default V = max(max(a),max(b))+1\n\
OUTPUT:\n\
- A,B,D as required \n\
";

static char graph_cc_doc[] = 
" label = graph_cc(a,b,d,V)\n\
  returns the connected components as labels.\n\
  the graph is assumed symmetric \n\
  INPUT:\n\
- The edges of the input graph are defined through the couple of 1-d arrays \n\
A,B such that [A[e] B[e]] are the vertices and D[e] an associated attribute \n\
(distance/weight/affinity) \n\
- V is the numner of vertices of the graph\n\
It is an optional argument, by default v = max(max(a),max(b))+1\n\
OUTPUT:\n\
- label is a length V label vector\n\
  ";

static char graph_is_connected_doc[] = 
" bool = graph_is_connected(a,b,d,V)\n\
  states whether the given graph is connected or not.\n\
  the graph is assumed symmetric \n\
  INPUT:\n\
- The edges of the input graph are defined through the couple of 1-d arrays \n\
A,B such that [A[e] B[e]] are the vertices and D[e] an associated attribute \n\
(distance/weight/affinity) \n\
- V is the numner of vertices of the graph\n\
It is an optional argument, by default v = max(max(a),max(b))+1\n\
OUTPUT:\n\
- bool, the response \n\
  ";

static char graph_mcc_doc[] = 
" idx = graph_main_cc(a,b,d,V)\n\
  returns the main connected component of the graph.\n\
  the graph is assumed symmetric \n\
 INPUT:\n\
- The edges of the input graph are defined through the couple of 1-d arrays \n\
A,B such that [A[e] B[e]] are the vertices and D[e] an associated attribute \n\
(distance/weight/affinity) \n\
- V is the numner of vertices of the graph\n\
It is an optional argument, by default v = max(max(a),max(b))+1\n\
OUTPUT:\n\
- an index array idx that contains the vertices of the largest cc\n\
  ";

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


static char graph_bpmatch_doc[] = 
" belief = graph_bpmatch(sources,target, adjacency,d0) \n\
   estimation of a probabilistic matching between source and target. \n\
   The source is defined as the rows of the source matrix.\n\
   The targets are defined as the rows of the target matrix \n\
   the algorithm uses a belief propagation algorithm between the sources \n\
   in oder to impose similar relative positions between sources and targets \n\
   The BP algo builds on the graph striucture which is codde \n\
   by an adjacency matrix adjacency \n\
   d0 is a cutoff/scale parameter \n\
INPUT:\n\
- sources: the position of the sources \n\
- targets: the posistion of the targets\n\
- adjacency: adjacency matrix of the graph structure between sources \n\
sources,targets and adjacency are arrays of size (n1,p),(n2,p) and (n1,n1) \n\
adjacency is expected to be symmetric \n\
OUTPUT: \n\
- belief: the probabilistic correspondence matrix; size (n1,n2)\n\
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
 ************ Part 1 : creating graphs **************************
 ***************************************************************/



static PyObject* graph_complete(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *d;
  int v,E;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "i:graph_complete", 
			     &v); 
  if (!OK) Py_RETURN_NONE; 
  
  /* prepare C arguments */
  fff_graph *G = fff_graph_complete(v);
  /* do the job */
  E = v*v; 
  fff_array *A = fff_array_new1d(FFF_LONG,E);
  fff_array *B = fff_array_new1d(FFF_LONG,E);
  fff_vector *D = fff_vector_new(E);

  fff_graph_edit_safe(A,B,D,G);

  fff_graph_delete(G);

  /* get the results as python arrrays*/
  a = fff_array_toPyArray( A );
  b =  fff_array_toPyArray( B );
  d = fff_vector_toPyArray( D );
  /* Output tuple */
  //  NNN = (tuple of) three Numpa Arrays. For other type, see doc at
  //  http://www.python.org/doc/2.4/ext/buildValue.html
  PyObject* ret = Py_BuildValue("NNN",  
				a, 
				b,
				d); 
  
  return ret;
}

static PyObject* graph_knn(PyObject* self, PyObject* args)
{

  PyArrayObject *x, *a, *b, *d;
  int E,k;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!i:graph_knn", 
			  &PyArray_Type, &x, 
			  &k); 
    if (!OK) Py_RETURN_NONE; 

  /* prepare C arguments */
  fff_matrix* X = fff_matrix_fromPyArray( x ); 
  fff_graph *G;

  /* do the job */
  E = fff_graph_knn(&G, X, k); 

  fff_array *A = fff_array_new1d(FFF_LONG,E);
  fff_array *B = fff_array_new1d(FFF_LONG,E);
  fff_vector *D = fff_vector_new(E);

  fff_graph_edit_safe(A,B,D,G);
  fff_graph_delete(G);
  fff_matrix_delete(X);

  /* get the results as python arrrays*/
  a = fff_array_toPyArray( A );
  b =  fff_array_toPyArray( B );
  d = fff_vector_toPyArray( D );

  /* Output tuple */
  PyObject* ret = Py_BuildValue("NNN", 
				a, 
				b,
				d); 

  return ret;
}

static PyObject* graph_cross_knn(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *y, *a, *b, *d;
  int k;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!i:graph_crossknn", 
			  &PyArray_Type, &x, 
			  &PyArray_Type, &y, 
			  &k); 
  if (!OK) Py_RETURN_NONE; 
 

  /* prepare C arguments */
  fff_matrix* X = fff_matrix_fromPyArray( x ); 
  fff_matrix* Y = fff_matrix_fromPyArray( y ); 
  int V = X->size1; 
  int E = k*V;
  fff_graph *G = fff_graph_new(V,E);
  fff_array *A = fff_array_new1d(FFF_LONG,E);
  fff_array *B = fff_array_new1d(FFF_LONG,E);
  fff_vector *D = fff_vector_new(E);
  
  /* do the job */
  E = fff_graph_cross_knn(G, X, Y, k);  
  fff_graph_edit_safe(A,B,D,G);
  fff_graph_delete(G);
  fff_matrix_delete(X);
  fff_matrix_delete(Y);

  /* get the results as python arrrays*/
  a = fff_array_toPyArray( A );
  b = fff_array_toPyArray( B );
  d = fff_vector_toPyArray( D );
  

  /* Output tuple */
  PyObject* ret = Py_BuildValue("NNN", 
				a, 
				b,
				d); 
  
  return ret;
}

static PyObject* graph_eps(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *a, *b, *d;
  int E;
  double eps;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!d:graph_eps", 
			  &PyArray_Type, &x, 
			  &eps); 
  if (!OK) Py_RETURN_NONE; 
  
  /* prepare C arguments */
  fff_matrix* X = fff_matrix_fromPyArray( x ); 
  fff_graph *G;
  
  /* do the job */
  E = fff_graph_eps(&G, X, eps); 
  fff_array *A = fff_array_new1d(FFF_LONG,E);
  fff_array *B = fff_array_new1d(FFF_LONG,E);
  fff_vector *D = fff_vector_new(E);
  
  fff_graph_edit_safe(A,B,D,G);
  fff_graph_delete(G);
  fff_matrix_delete(X);

  /* get the results as python arrrays*/
  a = fff_array_toPyArray( A );
  b =  fff_array_toPyArray( B );
  d = fff_vector_toPyArray( D );
  
  /* Output tuple */
  PyObject* ret = Py_BuildValue("NNN", 
				a, 
				b,
				d); 
  
  return ret;
}

static PyObject* graph_cross_eps(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *y, *a, *b, *d;
  int E;
  double eps;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!d:graph_cross_eps", 
			  &PyArray_Type, &x, 
			  &PyArray_Type, &y,
			  &eps); 
  if (!OK) Py_RETURN_NONE; 
  
  /* prepare C arguments */
  fff_matrix* X = fff_matrix_fromPyArray( x );
  fff_matrix* Y = fff_matrix_fromPyArray( y );
  fff_graph *G;
  
  /* do the job */
  E = fff_graph_cross_eps(&G, X, Y, eps); 
  fff_array *A = fff_array_new1d(FFF_LONG,E);
  fff_array *B = fff_array_new1d(FFF_LONG,E);
  fff_vector *D = fff_vector_new(E);
  
  fff_graph_edit_safe(A,B,D,G);
  fff_graph_delete(G);
  fff_matrix_delete(X);
  fff_matrix_delete(Y);

  /* get the results as python arrrays*/
  a = fff_array_toPyArray( A );
  b =  fff_array_toPyArray( B );
  d = fff_vector_toPyArray( D );
  
  /* Output tuple */
  PyObject* ret = Py_BuildValue("NNN", 
				a, 
				b,
				d); 
  
  return ret;
}

static PyObject* graph_cross_eps_robust(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *y, *a, *b, *d;
  int E;
  double eps;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!d:graph_cross_eps_robust", 
			  &PyArray_Type, &x, 
			  &PyArray_Type, &y,
			  &eps); 
  if (!OK) Py_RETURN_NONE; 
  
  /* prepare C arguments */
  fff_matrix* X = fff_matrix_fromPyArray( x );
  fff_matrix* Y = fff_matrix_fromPyArray( y );
  fff_graph *G;
  
  /* do the job */
  E = fff_graph_cross_eps_robust(&G, X, Y, eps); 
  fff_array *A = fff_array_new1d(FFF_LONG,E);
  fff_array *B = fff_array_new1d(FFF_LONG,E);
  fff_vector *D = fff_vector_new(E);
  
  fff_graph_edit_safe(A,B,D,G);
  fff_graph_delete(G);
  fff_matrix_delete(X);
  fff_matrix_delete(Y);

  /* get the results as python arrrays*/
  a = fff_array_toPyArray( A );
  b =  fff_array_toPyArray( B );
  d = fff_vector_toPyArray( D );
  
  /* Output tuple */
  PyObject* ret = Py_BuildValue("NNN", 
				a, 
				b,
				d); 
  
  return ret;
}

static PyObject* graph_mst(PyObject* self, PyObject* args)
{
  PyArrayObject *x, *a, *b, *d;
  
  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!:graph_mst", 
			  &PyArray_Type, &x); 
  if (!OK) Py_RETURN_NONE; 
  
  /* prepare C arguments */
  fff_matrix* X = fff_matrix_fromPyArray( x );
  int V = X->size1; 
  int E = 2*(V-1);
  fff_graph *G = fff_graph_new(V,E);
  fff_array *A = fff_array_new1d(FFF_LONG,E);
  fff_array *B = fff_array_new1d(FFF_LONG,E);
  fff_vector *D = fff_vector_new(E);

  /* do the job */
  fff_graph_MST(G, X);   
  fff_graph_edit_safe(A,B,D,G);
  fff_graph_delete(G);
  fff_matrix_delete(X);

  /* get the results as python arrrays*/
  a = fff_array_toPyArray( A );
  b =  fff_array_toPyArray( B );
  d = fff_vector_toPyArray( D );
  
  /* Output tuple */
  PyObject* ret = Py_BuildValue("NNN", 
				a, 
				b,
				d); 
  
  return ret;
}

static PyObject* graph_skeleton(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *d;
  int V;
  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!|i:graph_skeleton", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &d,
							 &V); 
  if (!OK) Py_RETURN_NONE; 
  
  /* prepare C arguments */
  fff_array* A = fff_array_fromPyArray( a ); 
  fff_array* B = fff_array_fromPyArray( b );
  fff_vector* D = fff_vector_fromPyArray(d);
  int E = A->dimX;

  /* do the job */
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  
  fff_array_delete(A);
  fff_array_delete(B);
  fff_vector_delete(D);

  E = 2*(V-1);
  fff_graph *K = fff_graph_new(V,E);

 /* do the job */
  fff_graph_skeleton(K, G);   
  A = fff_array_new1d(FFF_LONG,E);
  B = fff_array_new1d(FFF_LONG,E);
  D = fff_vector_new(E);  
  
  fff_graph_edit_safe(A,B,D,K);
  fff_graph_delete(G);
  fff_graph_delete(K);

  /* get the results as python arrrays*/
  a = fff_array_toPyArray( A );
  b =  fff_array_toPyArray( B );
  d = fff_vector_toPyArray( D );
  
  /* Output tuple */
  PyObject* ret = Py_BuildValue("NNN", 
				a, 
				b,
				d); 
  
  return ret;
}

static PyObject* graph_3d_grid(PyObject* self, PyObject* args)
{
  PyArrayObject *xyz, *a, *b, *d;
  int E,k=18;

  /* Parse input */ 
  
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!|i:graph_3d_grid", 
			  &PyArray_Type, &xyz, 
			  &k); 
  if (!OK) Py_RETURN_NONE; 
  
  fff_array* XYZ = fff_array_fromPyArray( xyz );   
  fff_graph *G;

  E = fff_graph_grid(&G,XYZ, k);
  
  if (E == -1) {
      FFF_WARNING("Graph creation failed");
      Py_RETURN_NONE;
  }


  fff_array_delete(XYZ);

  fff_array *A = fff_array_new1d(FFF_LONG,E);
  fff_array *B = fff_array_new1d(FFF_LONG,E);
  fff_vector *D = fff_vector_new(E);

  fff_graph_edit_safe(A,B,D,G);
  fff_graph_delete(G);
  /* get the results as python arrrays*/
  a = fff_array_toPyArray( A );
  b = fff_array_toPyArray( B );
  d = fff_vector_toPyArray( D );
  
  /* Output tuple */
  PyObject* ret = Py_BuildValue("NNN", 
				a, 
				b,
				d); 
  
  return ret;
} 

/****************************************************************
 ************ Part 2 : graph analysis ***************************
 ***************************************************************/


static PyObject* graph_degrees(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *rd, *ld;
  int E,eA,eB,V=0;
  

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!|i:graph_degrees", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &V
			     ); 
  if (!OK) Py_RETURN_NONE; 
    
  /* prepare C arguments */
  fff_array* A = fff_array_fromPyArray( a ); 
  fff_array* B = fff_array_fromPyArray( b );
  E = A->dimX;
  if (V<1){
    eA = (int)_fff_array_max1d(A)+1;
    eB = (int)_fff_array_max1d(B)+1;
    if (eA>V) V = eA;
    if (eB>V) V = eB;
  }
  fff_vector* D = fff_vector_new(E);
  fff_array* Rd = fff_array_new1d(FFF_LONG,V);
  fff_array* Ld = fff_array_new1d(FFF_LONG,V);
  
  
  /* do the job */
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  if (G==NULL){
    Rd = NULL;
    Ld = NULL;
  }
  else{
    fff_graph_ldegrees( Ld->data, G);
    fff_graph_rdegrees( Rd->data, G);
  }

  /* free the memory */
  fff_graph_delete(G);
  fff_vector_delete(D);
  fff_array_delete(A);
  fff_array_delete(B);
  
  /* get the results as python arrrays*/
  rd = fff_array_toPyArray( Rd );
  ld = fff_array_toPyArray( Ld );
  
  /* Output tuple */
  PyObject* ret = Py_BuildValue("NN", rd, ld); 
  return ret;
}



static PyArrayObject* graph_adjacency(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *d, *m;
  int eA,eB,V=0;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!|i:graph_adjacency", 
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
  fff_matrix *M;
  if (V<1){
    eA = (int)_fff_array_max1d(A)+1;
    eB = (int)_fff_array_max1d(B)+1;
    if (eA>V) V = eA;
    if (eB>V) V = eB;
  }
  
  /* do the job */
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  fff_graph_to_matrix(&M,G);
  fff_graph_delete(G);
  fff_array_delete(A);
  fff_array_delete(B);
  fff_vector_delete(D);

  /* get the results as python arrrays*/
  m = fff_matrix_toPyArray( M );
  
  return m;
}

static PyObject* graph_to_neighb(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *d, *ci, *ne, *we;
  int eA,eB,V=0;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!i:graph_to_neighb", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &d,
			     &V
			     ); 
  if (!OK) Py_RETURN_NONE; 
    
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

  fff_array* cindices = fff_array_new1d(FFF_LONG, V+1 );
  fff_array* neighb = fff_array_new1d(FFF_LONG, E );
  fff_vector* weight = fff_vector_new( E );

  fff_graph_to_neighb(cindices, neighb, weight, G);

  fff_graph_delete(G);
  
  /* get the results as python arrrays*/
  ci = fff_array_toPyArray( cindices );
  ne = fff_array_toPyArray( neighb );
  we = fff_vector_toPyArray( weight );

  /* Output tuple */
  PyObject* ret = Py_BuildValue("NNN", 
				ci,
				ne,
				we); 
  return ret;
}


static PyObject* graph_symmeterize(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *d;
  int eA,eB,V=0;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!|i:graph_symmeterize", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &d,
			     &V
			     ); 
  if (!OK) Py_RETURN_NONE; 
    
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
  fff_graph *K; 
  E = fff_graph_symmeterize(&K,G);
  fff_graph_delete(G);
  
  A = fff_array_new1d(FFF_LONG, E );
  B = fff_array_new1d(FFF_LONG, E );
  D = fff_vector_new( E );
  fff_graph_edit_safe(A,B,D,K);
  fff_graph_delete(K);
  
  /* get the results as python arrrays*/
  a = fff_array_toPyArray( A );
  b = fff_array_toPyArray( B );
  d = fff_vector_toPyArray( D );

  /* Output tuple */
  PyObject* ret = Py_BuildValue("NNN", 
				a,
				b,
				d); 
  return ret;
}

static PyObject* graph_antisymmeterize(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *d;
  int eA,eB,V=0;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!|i:graph_antisymmeterize", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &d,
			     &V
			     ); 
  if (!OK) Py_RETURN_NONE; 
    
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
  fff_graph *K; 
  E = fff_graph_antisymmeterize(&K,G);
  fff_graph_delete(G);
  
  A = fff_array_new1d(FFF_LONG, E );
  B = fff_array_new1d(FFF_LONG, E );
  D = fff_vector_new( E );
  fff_graph_edit_safe(A,B,D,K);
  fff_graph_delete(K);
  
  /* get the results as python arrrays*/
  a = fff_array_toPyArray( A );
  b = fff_array_toPyArray( B );
  d = fff_vector_toPyArray( D );

  /* Output tuple */
  PyObject* ret = Py_BuildValue("NNN", 
				a,
				b,
				d); 
  return ret;
}


static PyArrayObject* graph_set_euclidian(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *d, *x;
  
  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!:graph_set_euclidian",
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &x); 
  if (!OK) return NULL;  
    
  /* prepare C arguments */
  fff_array* A = fff_array_fromPyArray( a ); 
  fff_array* B = fff_array_fromPyArray( b );  
  fff_matrix* X = fff_matrix_fromPyArray(x);
  int V = X->size1;
  int E = A->dimX;
  fff_vector* D = fff_vector_new(E);
  fff_vector_set_all(D,0);

  /* do the job */
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  fff_graph_set_euclidian(G, X);
  fff_graph_edit_safe(A,B,D,G);
  fff_graph_delete(G);
  fff_matrix_delete(X);
  fff_array_delete(A);
  fff_array_delete(B);

  /* get the results as python arrrays*/
  d = fff_vector_toPyArray( D );

  return d;
}

static PyArrayObject* graph_set_gaussian(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *d, *x;
  double sigma = 0;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!|d:graph_set_gaussian", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &x,
			     &sigma
                            ); 
  if (!OK) return NULL; 
    
  /* prepare C arguments */
  fff_array* A = fff_array_fromPyArray( a ); 
  fff_array* B = fff_array_fromPyArray( b );
  fff_matrix* X = fff_matrix_fromPyArray(x);
  int E = A->dimX;
  int V = X->size1;
  fff_vector* D = fff_vector_new(E);
  fff_vector_set_all(D,0);
  
  /* do the job */
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  if (sigma>0)
    fff_graph_set_Gaussian(G, X, sigma);
  else
    fff_graph_auto_Gaussian(G, X);

  fff_graph_edit_safe(A,B,D,G);
  fff_graph_delete(G);
  fff_matrix_delete(X);
  fff_array_delete(A);
  fff_array_delete(B);

  /* get the results as python arrrays*/
  d = fff_vector_toPyArray( D );

  return d;
}

static PyObject* graph_reorder(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *d;
  int eA,eB,V = 0;
  int c = 0;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!|ii:graph_reorder", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &d,
			     &c,
                             &V); 
  if (!OK) Py_RETURN_NONE; 
    
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
    
  if (c>2) c=0;
  switch (c)
    {
    case 0:{
      fff_graph_reorderA(G);
      break;}
    case 1:{
      fff_graph_reorderB(G);
      break;}
    case 2:{
      fff_graph_reorderD(G);
      break;}
    }
  fff_graph_edit_safe(A,B,D,G);
  fff_graph_delete(G);
  
  /* get the results as python arrrays*/
  a = fff_array_toPyArray( A );
  b = fff_array_toPyArray( B );
  d = fff_vector_toPyArray( D );

  /* Output tuple */
  PyObject* ret = Py_BuildValue("NNN", 
				a,
				b,
				d); 
  return ret;
}

static PyObject* graph_normalize(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *d, *s, *t;
  int eA,eB,V=0;
  int c=0;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!|ii:graph_normalize", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &d,
			     &c,			     
                             &V
                             ); 
  if (!OK) Py_RETURN_NONE; 
    
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
  fff_vector* S = fff_vector_new(V);
  fff_vector* T = NULL;
  
  if (c>2) c=0;

  switch (c)
    {
    case 0:{
      fff_graph_normalize_rows(G,S);
      break;}
    case 1:{
      fff_graph_normalize_columns(G,S);
      break;}
	case 2:{
	  T = fff_vector_new(V);
      fff_graph_normalize_symmetric(G,S,T);
      break;}
    }
  fff_graph_edit_safe(A,B,D,G);
  fff_graph_delete(G);
  
  /* get the results as python arrrays*/
  s = fff_vector_toPyArray( S );
  a = fff_array_toPyArray( A );
  b = fff_array_toPyArray( B );
  d = fff_vector_toPyArray( D );

  /* Output tuple */
  PyObject* ret;
  if (c<2)ret = Py_BuildValue("NNNN",a,b,d,s);
  else{
	t = fff_vector_toPyArray( T );
	ret = Py_BuildValue("NNNNN",a,b,d,s,t);
  }
	
  return ret;
}


static PyObject* graph_cut_redundancies(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *d;
  int eA,eB,V=0;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!|i:graph_cut_redundancies", 
			     &PyArray_Type, &a,
			     &PyArray_Type, &b,
			     &PyArray_Type, &d,
                             &V
                             ); 
  
  if (!OK) Py_RETURN_NONE; 
    
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
  fff_graph *K;
  fff_graph_cut_redundancies(&K, G);
    E = K->E;
  fff_array_delete(A);
  fff_array_delete(B);
  fff_vector_delete(D);
  
  A = fff_array_new1d(FFF_LONG,E);
  B = fff_array_new1d(FFF_LONG,E);
  D = fff_vector_new(E);
  
  fff_graph_edit_safe(A,B,D,K);
  fff_graph_delete(G);
  fff_graph_delete(K);
      
  /* get the results as python arrrays*/
  a = fff_array_toPyArray( A );
  b = fff_array_toPyArray( B );
  d = fff_vector_toPyArray( D );

  /* Output tuple */
  PyObject* ret = Py_BuildValue("NNN", 
				a,
				b,
				d); 
  return ret;
}

static PyObject* graph_is_connected(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *d;
  int eA,eB,V=0;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!|i:graph_is_connected", 
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
  int bool = fff_graph_isconnected(G);
   fff_graph_delete(G);
  
  PyObject *ret = Py_BuildValue("i",bool);
  return ret;
}

static PyArrayObject* graph_cc(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *d, *l;
  int eA,eB,V=0;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!|i:graph_cc", 
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
  fff_array *label = fff_array_new1d(FFF_LONG,V);
  
  /* do the job */
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  fff_array_delete(A);
  fff_array_delete(B);
  fff_vector_delete(D);
  
  fff_graph_cc_label(label->data,G);
  fff_graph_delete(G);
  
  /* get the results as python arrrays*/
  l = fff_array_toPyArray( label );
  
  return l;
}

static PyArrayObject* graph_main_cc(PyObject* self, PyObject* args)
{
  PyArrayObject *a, *b, *d, *m;
  int eA,eB,V=0;

  /* Parse input */ 
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html*/
  int OK = PyArg_ParseTuple( args, "O!O!O!|i:graph_main_cc", 
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
  fff_array *mcc;
  
  /* do the job */
  fff_graph *G = fff_graph_build_safe(V,E,A,B,D);
  fff_array_delete(A);
  fff_array_delete(B);
  fff_vector_delete(D);
  
  fff_graph_main_cc(&mcc,G);
  fff_graph_delete(G);
  
  /* get the results as python arrrays*/
  m = fff_array_toPyArray( mcc );

  return m;
}

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


static PyArrayObject* graph_bpmatch(PyObject* self, PyObject* args)
{
  PyArrayObject *t, *b, *s, *g;
  double d;

  /* Parse input */
  /* see http://www.python.org/doc/1.5.2p2/ext/parseTuple.html */
  int OK = PyArg_ParseTuple( args, "O!O!O!d:graph_bpmatch", 
			     &PyArray_Type, &s,
			     &PyArray_Type, &t,
			     &PyArray_Type, &g,
			     &d
                             ); 
  if (!OK) return NULL;   
    
  /* prepare C arguments */
  fff_matrix* source = fff_matrix_fromPyArray( s ); 
  fff_matrix* target = fff_matrix_fromPyArray( t );
  fff_matrix* graph = fff_matrix_fromPyArray( g );

  fff_matrix *belief = fff_matrix_new(source->size1,target->size1);
  
  fff_BPmatch(source, target, graph, belief, d);
  
  /* get the results as python arrrays */
  b = fff_matrix_toPyArray( belief );

  /* Output tuple  */
  return b;
}



static PyMethodDef module_methods[] = {
   {"graph_complete",           /* name of func when called from Python */
   (PyCFunction)graph_complete,             /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_complete_doc},        /* doc string */
   {"graph_knn",           /* name of func when called from Python */
   (PyCFunction)graph_knn,             /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_knn_doc},        /* doc string */
  {"graph_eps",           /* name of func when called from Python */
   (PyCFunction)graph_eps,             /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_eps_doc},        /* doc string */
   {"graph_cross_knn",    /* name of func when called from Python */
   (PyCFunction)graph_cross_knn,       /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_cross_knn_doc},  /* doc string */
  {"graph_cross_eps",     /* name of func when called from Python */
   (PyCFunction)graph_cross_eps,       /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_cross_eps_doc},  /* doc string */
  {"graph_cross_eps_robust",     /* name of func when called from Python */
   (PyCFunction)graph_cross_eps_robust,       /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_cross_eps_robust_doc},  /* doc string */
  {"graph_mst",           /* name of func when called from Python */
   (PyCFunction)graph_mst,             /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_mst_doc},        /* doc string */
  {"graph_3d_grid",        /* name of func when called from Python */
   (PyCFunction)graph_3d_grid,          /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_3d_grid_doc},        /* doc string */
  {"graph_degrees",        /* name of func when called from Python */
   (PyCFunction)graph_degrees,          /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_degrees_doc},        /* doc string */
  {"graph_adjacency",        /* name of func when called from Python */
   (PyCFunction)graph_adjacency,          /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_adjacency_doc},        /* doc string */
   {"graph_to_neighb",        /* name of func when called from Python */
   (PyCFunction)graph_to_neighb,          /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_to_neighb_doc},        /* doc string */
   {"graph_symmeterize",        /* name of func when called from Python */
   (PyCFunction)graph_symmeterize,          /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_symmeterize_doc},        /* doc string */
   {"graph_antisymmeterize",        /* name of func when called from Python */
   (PyCFunction)graph_antisymmeterize,          /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_antisymmeterize_doc},        /* doc string */
   {"graph_reorder",        /* name of func when called from Python */
   (PyCFunction)graph_reorder,          /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_reorder_doc},        /* doc string */
   {"graph_normalize",        /* name of func when called from Python */
   (PyCFunction)graph_normalize,          /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_normalize_doc},        /* doc string */
   {"graph_cut_redundancies",        /* name of func when called from Python */
   (PyCFunction)graph_cut_redundancies,          /* corresponding C function */
    METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_cut_redundancies_doc},        /* doc string */
   {"graph_set_gaussian",        /* name of func when called from Python */
   (PyCFunction)graph_set_gaussian,          /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_set_gaussian_doc},        /* doc string */
  {"graph_set_euclidian",        /* name of func when called from Python */
   (PyCFunction)graph_set_euclidian,          /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_set_euclidian_doc},        /* doc string */
  {"graph_cc",        /* name of func when called from Python */
   (PyCFunction)graph_cc,          /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_cc_doc},        /* doc string */
  {"graph_main_cc",        /* name of func when called from Python */
   (PyCFunction)graph_main_cc,          /* corresponding C function */
   METH_KEYWORDS,          /* ordinary (not keyword) arguments */
   graph_mcc_doc},        /* doc string */
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
   {"graph_bpmatch",        /* name of func when called from Python */
	(PyCFunction)graph_bpmatch,          /* corresponding C function */
	METH_KEYWORDS,          /*ordinary (not keyword) arguments */
	graph_bpmatch_doc},        /* doc string */
   {"graph_skeleton",        /* name of func when called from Python */
	(PyCFunction)graph_skeleton,          /* corresponding C function */
	METH_KEYWORDS,          /*ordinary (not keyword) arguments */
	graph_skeleton_doc},        /* doc string */
    {"graph_is_connected",        /* name of func when called from Python */
	 (PyCFunction)graph_is_connected,          /* corresponding C function */
	METH_KEYWORDS,          /*ordinary (not keyword) arguments */
	graph_is_connected_doc},        /* doc string */
   {NULL, NULL,0,NULL}

};


void init_graph(void)
{
  Py_InitModule3("_graph", module_methods, module_doc);
  fffpy_import_array();
  import_array();   /* required NumPy initialization */
}
