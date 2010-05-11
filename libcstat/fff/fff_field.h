/*!
  \file fff_field.h
  \brief Field processing functions.
  \author Bertrand Thirion
  \date 2006

  In this library, the term field stands for a function defined on a graph.
  More pecisely, given a graph, we consider function s defined on the vertices
  of the graph.
  The graph structure induces some operations on the values:
  diffusion (smoothing), morphology, local optima ...
  From an implementation point of view, we deal with the two elements 
  (graph and function) separately, though a structure ould be introduced.

  */

#ifndef fff_FIELD
#define fff_FIELD

#ifdef __cplusplus
extern "C" {
#endif

#include "fff_graphlib.h"
#include "fff_array.h"
#include "fff_vector.h"
#include "fff_matrix.h"  
#include "fff_blas.h"

/*!
    \brief returns the local maxima of the function field defined on the graph
    \param depth topological depth of the local maxima (0 for non-maxima)
    \param G  graph
    \param field  values on the vertices of the graph
    \param rec number of iterations of the algorithm (it is also the maximal depth)

    Given a graphG (V vertices), and a function defined on [1..V], the function
    coputes the depth de[v] of each vertex so that:
    forall w in N(..N(N(v))), field(v)>=field(w), where the recursion
    is performed depth[v] times, and N(.) stands for the neighborhood 
    of a vertex
    in the graph topology.
    depth[v] = 0 for non-maxima
    0=<depth[v]<rec
  */
  extern int fff_field_maxima_r(fff_array *depth, const fff_graph* G, const fff_vector* field,  const int rec);

/*!
    \brief returns the local maxima of the function field defined on the graph
    \param depth topological depth of the local maxima (0 for non-maxima)
    \param G  graph
    \param field  values on the vertices of the graph
    
    Given a graphG (V vertices), and a function defined on [1..V], the function
    coputes the depth de[v] of each vertex so that:
    forall w in N(..(N(v))), field(v)>=field(w), where the recursion
    is performed depth[v] times, and N(.) stands for the neighborhood 
    of a vertex
    in the graph topology.
    depth[v] = 0 for non-maxima
    0=<depth[v]<=diam(G)
*/
  extern int fff_field_maxima(fff_array *depth, const fff_graph* G, const fff_vector *field);

  /*!
    \brief returns the local maxima of the function field defined on the graph
    \param depth topological depth of the local maxima 
    \param idx the indices of the local maxima
    \param G  graph
    \param field  values on the vertices of the graph
    
    This function basically calls fff_field_maxima. the difference is 
    that only the list of maxima is returned, together with the 
    associated depth
*/
  extern int fff_field_get_maxima(fff_array **depth, fff_array **idx,const fff_graph* G, const fff_vector *field);

  /*!
    \brief returns the supra-threshold local maxima of the function field defined on the graph
    \param depth topological depth of the local maxima 
    \param idx the indices of the local maxima
    \param G  graph
    \param field  values on the vertices of the graph
    \param th threshold above which maxima are considered
    
    This function basically calls fff_field_maxima. the difference is 
    that only the list of maxima is returned, together with the 
    associated depth
*/
  extern int fff_field_get_maxima_th(fff_array **depth, fff_array ** idx,const fff_graph* G, const fff_vector *field, const double th);


/*!
    \brief returns the local minima of the function field defined on the graph
    \param depth topological depth of the local minima (0 for non-minima)
    \param G  graph
    \param field  values on the vertices of the graph
    \param rec number of iterations of the algorithm (it is also the minimal depth)

    Given a graphG (V vertices), and a function defined on [1..V], the function
    coputes the depth de[v] of each vertex so that:
    forall w in N(..N(N(v))), field(v)>=field(w), where the recursion
    is performed depth[v] times, and N(.) stands for the neighborhood 
    of a vertex
    in the graph topology.
    depth[v] = 0 for non-minima
    0=<depth[v]<rec
  */
  extern int fff_field_minima_r(fff_array *depth, const fff_graph* G, const fff_vector* field,  const int rec);

/*!
    \brief returns the local minima of the function field defined on the graph
    \param depth topological depth of the local minima (0 for non-minima)
    \param G  graph
    \param field  values on the vertices of the graph
    
    Given a graphG (V vertices), and a function defined on [1..V], the function
    coputes the depth de[v] of each vertex so that:
    forall w in N(..(N(v))), field(v)>=field(w), where the recursion
    is performed depth[v] times, and N(.) stands for the neighborhood 
    of a vertex
    in the graph topology.
    depth[v] = 0 for non-minima
    0=<depth[v]<=diam(G)
*/
  extern int fff_field_minima(fff_array *depth, const fff_graph* G, const fff_vector *field);

  /*!
    \brief returns the local minima of the function field defined on the graph
    \param depth topological depth of the local minima 
    \param idx the indices of the local minima
    \param G  graph
    \param field  values on the vertices of the graph
    
    This function basically calls fff_field_minima. the difference is 
    that only the list of minima is returned, together with the 
    associated depth
*/
  extern int fff_field_get_minima(fff_array **depth, fff_array **idx,const fff_graph* G, const fff_vector *field);

  /*!
    \brief sparse kernel diffusion
    \param field field of data that is diffused
    \param G  graph
    
    Interpreting the graph G as a sparse kernel, the algorithm
    performs one iteration of diffusion on the field data.
  */
  extern int fff_field_diffusion(fff_vector *field, const fff_graph* G);
  
  /*!
    \brief sparse kernel diffusion
    \param field field of data that is diffused
    \param G  graph
    \
    
    Interpreting the graph G as a sparse kernel, the algorithm
    performs one iteration of diffusion on the field data.
    the firld is multi-dimensional, hence a matrix
  */
  extern int fff_field_md_diffusion(fff_matrix *field, const fff_graph* G);

    /*!
    \brief morphological dilation of the field of 1 unit
    \param field field of data that is diffused
    \param G  graph
    \param rec (topological) radius of the dilation

    Interpreting the graph G as a sparse kernel, the algorithm
    performs one iteration of dilation on the field data.
  */
  extern int fff_field_dilation(fff_vector *field, const fff_graph* G, const int rec);

  /*!
    \brief morphological erosion of the field of rec unit
    \param field field of data that is diffused
    \param G  graph
    \param rec (topological) radius of the erosion
    
    Interpreting the graph G as a sparse kernel, the algorithm
    performs an erosion of radius rec on the field data.
  */
  extern int fff_field_erosion(fff_vector *field, const fff_graph* G, const int rec);

/*!
    \brief morphological opening of the field of rec unit
    \param field field of data that is diffused
    \param G  graph
    \param rec (topological) radius of the opening
    
    Interpreting the graph G as a sparse kernel, the algorithm
    performs an openeing of radius rec on the field data.
  */
  extern int fff_field_opening(fff_vector *field, const fff_graph* G, const int rec);

/*!
    \brief morphological closing of the field of rec unit
    \param field field of data that is diffused
    \param G  graph
    \param rec (topological) radius of the closing
    
    Interpreting the graph G as a sparse kernel, the algorithm
    performs a closing of radius rec on the field data.
  */
  extern int fff_field_closing(fff_vector *field, const fff_graph* G, const int rec);

/*!
    \brief customized watershed analysis of the field
    \param idx gives the indices of the maxima in each bassin
    \param depth gives the (topological) depth of these maxima
    \param major gives the index of the nearest dominating maximum
    \param label is a labelling of the vertices according to the watershed bassins
    \param field field of data that is diffused
    \param G  graph
    
    The number q of bassins is returned.
    the first three vectors (idx,depth,major) are of size q.
    Label is of size field->size
    Note that bassins are defined as zones around maxima, 
    unlike the usual intuition.
  */
  extern int fff_custom_watershed(fff_array **idx, fff_array **depth, fff_array **major, fff_array *label, const fff_vector *field, const fff_graph* G);

/*!
    \brief customized watershed analysis of the field, with only supra-threshold parts considered
    \param idx gives the indices of the maxima in each bassin
    \param depth gives the (topological) depth of these maxima
    \param major gives the index of the nearest dominating maximum
    \param label is a labelling of the vertices according to the watershed bassins
    \param field field of data that is diffused
    \param G  graph
    \param th threshold
    
    The number q of bassins is returned.
    the first three vectors (idx,depth,major) are of size q.
    Label is of size field->size
    Note that bassins are defined as zones around maxima, 
    unlike the usual intuition.
  */
  extern int fff_custom_watershed_th(fff_array **idx, fff_array **depth, fff_array **major, fff_array *label, const fff_vector *field, const fff_graph* G, const double th);

  
/*!
    \brief customized watershed analysis of the field, with only supra-threshold parts considered
    \param idx gives the indices of the maxima in each bassin
    \param depth gives the (topological) depth of these maxima
    \param major gives the index of the nearest dominating maximum
    \param label is a labelling of the vertices according to the watershed bassins
    \param field field of data that is diffused
    \param G  graph
    \param th threshold
    
    The number q of bassins is returned.
    the first three vectors (idx,depth,major) are of size q.
    Label is of size field->size
    Note that bassins are defined as zones around maxima, 
    unlike the usual intuition.
  */
  extern long fff_field_bifurcations(fff_array **Idx, fff_vector **Height, fff_array **Father, fff_array* label,  const fff_vector *field, const fff_graph* G, const double th);

  /*!
    \brief Voronoi parcellation of the field structure, starting from given seed
	
	\param label is a label vector for the field
	\param G is the input graph
	\param field is the input data that drives the clsutering
	\param seeds are the seed points for the diferent labels

	This is simply a nearest-neighbour assignement, with a 'geodesic' constrin given by the graph 

  */
  extern long fff_field_voronoi(fff_array *label, const fff_graph* G,const fff_matrix* field,const  fff_array *seeds);

#ifdef __cplusplus
}
#endif

#endif
