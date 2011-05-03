/*!
  \file fff_graphlib.h
  \brief Graph modelling and processing functions.
  \author Bertrand Thirion
  \date 2004-2006

  This library implements different low- and high-level functions for 
  graph processing.

  low-level functions include: graph "constructor" and "destructor",
  labelling of the vertices according to the connected component they
  belong to, extraction of the main cc, computation of the edge
  degrees. Note that the vertices are referred to as [0..V-1], V being the
  number of vertices
  
  Higher level functions include Construction of the k nearest neighbors, 
  epsilon neighbours graph, Minimum Spanning Tree. For these functions, the 
  dataset is given as a matrix, assuming an implicit Euclidian distance.
  
  Last, Dijkstra's and Floyd's algorithm have been implemented.

  An important choice is that the graph is represented by a sparse
  adjacency matrix coding. In the current state of the structure, the
  later is a 3 vectors structure (A,B,D). E being the number of edges, an
  edge i, i<E,  is defined as the directed [A(i) B(i)] segment; D(i) is 
  a value associated with the edge (e.g. a length or a weight).
  
  The coding is appropriate for very large numbers of vertices with
  sparse connections. It is clearly suboptimal for small, dense
  graphs.
  
  2008/04/02: 
  To be implemented : 
  - quick list handling
  - a suboptimal part in MST
  */

#ifndef fff_GRAPHLIB
#define fff_GRAPHLIB

#ifdef __cplusplus
extern "C" {
#endif

#include "fff_array.h"
#include "fff_vector.h"
#include "fff_matrix.h"
#include "fff_base.h"


  typedef struct fff_graph{
    
    long V;                /*!< Number of vertices of the graph */
    long E;                /*!< Number of Edges of the graph */
    long* eA;                 /*!< edge origins (E) */
    long* eB;                 /*!< edge ends (E) */
    double* eD;              /*!< edge weights (E) */
    
  } fff_graph;

 /*!
    \struct fff_graph
    \brief Sparse graph structure
  */

  /*! 
    \brief Constructor for the fff_graph structure 
    \param v : number of vertices
    \param e : number of edges
  */
  extern fff_graph* fff_graph_new( const long v, const long e );
  /*! 
    \brief Destructor for the fff_graph structure 
    \param thisone the fff_graph structure to be deleted
  */
  extern void fff_graph_delete( fff_graph* thisone );

  /*! 
    \brief Other Constructor for the fff_graph structure 
    \param v the number of edges to be set
    \param e the number of vertices to be set
    \param A the origins of edges to be set
    \param B the ends of edges to be set
    \param D the values of edges to be set
  */
  extern fff_graph* fff_graph_build(const long v, const long e, const long *A, const long* B, const double*D );
    /*! 
    \brief Other Constructor for the fff_graph structure 
    \param v the number of edges to be set
    \param e the number of vertices to be set
    \param A the origins of edges to be set
    \param B the ends of edges to be set
    \param D the values of edges to be set
  */
  extern fff_graph* fff_graph_build_safe(const long v, const long e, const fff_array *A, const fff_array* B, const fff_vector *D );
 
   /*! 
    \brief edit the structure of a graph
    \param A the origins of edges
    \param B the ends of edges 
    \param D the values of edges 
    \param thisone the edited graph
  */
  extern void fff_graph_edit_safe(fff_array *A, fff_array* B, fff_vector *D, const fff_graph* thisone );
  
  /*
    \brief Conversion of a graph into a neighboring system
    \param cindices indexes of the neighbors of each vertex
    \param neighb neigbor list
    \param weight weight list
    \param G input graph

    this returns another sparse coding of the graph structure, in which each edge (eA[i],eB[i],eD[i])
    is coded as:
    for j in [cindices[a] cindices[a+1][, (a,eB[j],eD[j]) is an edge of G
    The advantage  is that the coding is sparser, and that the "neighbours of a" are directly given
    by the definition.
    cindices must be allocated G->V+1 elements
    neigh and weight must be allocated G->E elements
  */
  extern long fff_graph_to_neighb(fff_array *cindices, fff_array * neighb, fff_vector* weight, const fff_graph* G);
    

/*!
    \brief Minimum Spanning Tree construction from an existing graph
    \param G input graph
    \param K resulting sparse graph

    This algorithm builds a graph whose vertices are the list of items
    The number of edges is 2*nb vertices-2, due to the symmetry.
    The metric used in the algo is Euclidian.
    The algo used is Boruvska's algorithm. It is not fully optimized yet.

    The length of the MST or "skeleton" is returned
  */
  double fff_graph_skeleton(fff_graph* K,const fff_graph* G);

  /*!
    \brief graph labelling by connected components
    \param label resulting labels     
    \param G  sparse graph
 
    Given a graphG (V vertices),
    this algorithm builds a set of labels of size V, where each vertex 
    of one connected component of the graph has a given label
    It is assumed that label has been allocated enough size (G->V sizeof(double))
    It is assumed that the graph is undirected 
    (i.e. connectivity is assessed in the non-directed sense)

    the number of cc's is returned
  */
  extern long fff_graph_cc_label(long* label, const fff_graph* G);
  
  /*!
    \brief Dijkstra's algorithm
    \param dist the computed distance vector
    \param G  graph
    \param seed Dijkstra's algo seed point
    \param infdist infinite distance 
    
    Given a graph G, this algorithm compute Dijkstra's algo on 
    the weights of the graph.
    note that all the edge weights should be positive ! (distance graph)
    seed should be given in the interval 0,..,V-1
    infdist can be chosen typically as the sum of the edge weights
    of the graph.
  */
  extern long fff_graph_Dijkstra(double *dist, const fff_graph* G,const long seed, const double infdist );

/*!
    \brief Dijkstra's algorithm
    \param dist the computed distance vector
    \param G  graph
    \param seed Dijkstra's algo seed point
    
    Given a graph G, this algorithm compute Dijkstra's algo on 
    the weights of the graph. teh positivity of G->eD is checked.
    seed should be given in the interval 0,..,V-1
  */
  extern long fff_graph_dijkstra(double *dist, const fff_graph* G,const long seed);


  /*!
    \brief Dijkstra's algorithm
    \param dist the computed distance vector
    \param G  graph
    \param seeds Dijkstra's algo seed points
    
    Given a graph G, this algorithm compute Dijkstra's algo on 
    the weights of the graph. teh positivity of G->eD is checked.
    seeds should be given in the interval 0,..,V-1
	the null set is now an extended region
  */
  extern int fff_graph_Dijkstra_multiseed( fff_vector* dist, const fff_graph* G, const fff_array* seeds);

  /*!
    \brief Partial Floyd's algorithm
    \param dist the computed distance matrix (seeds*vertices)
    \param G  graph
    \param seeds the set of seed points from which geodesics are computed

    Given a graph G, this algorithm perform's a pseudo Floyd's algo on 
    the weights of the graph, by repetition of Dijkstra's algo
    from the seeds
    seeds should be given in the interval 0,..,V-1
    dist should be of size(nb(seeds),G->V)
  */
  extern long fff_graph_partial_Floyd(fff_matrix *dist, const fff_graph* G,const  long *seeds);

  /*!
    \brief Pseudo Floyd's algorithm
    \param dist the computed distance matrix (vertices*vertices)
    \param G  graph
    
    Given a graph G, this algorithm perform's a pseudo Floyd's algo on 
    the weights of the graph, by repetition of Dijkstra's algo
    from the vertices
    Note that all the edge weights should be positive ! (distance graph)
  */
  extern long fff_graph_Floyd( fff_matrix *dist, const fff_graph* G);

/*!
    \brief geodesic Voronoi algorithm
    \param label is the Voronoi vertices labelling 
    \param G  graph
    \param seeds the set of seed points of geodesic cells

    Given a graph G and seed points , this algorithm perform's a Voronoi
    labelling of the graph vertices, using the graph distance.
    Note that all the edge weights should be positive ! (distance graph)
  */
  extern long fff_graph_voronoi(fff_array *label, const fff_graph* G,const  fff_array *seeds);

  
/*!
    \brief Cliques extraction algorithm based on replicator dynamics
    \param cliques a labelling of the vertices according to the clique they belong to
    \param G  graph
    
    Given a graph with positive weights, this algo recursively 
    searches for the largest clique using RD framework.
    Note that due to stochastic initialization, the results may 
    vary from time to time
  */
  extern long fff_graph_cliques(fff_array *cliques, const fff_graph* G);


#ifdef __cplusplus
}
#endif

#endif
