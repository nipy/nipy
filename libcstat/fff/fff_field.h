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
