/*!
  \file fff_clustering.h
  \brief Data clustering functions.
  \author Bertrand Thirion
  \date 2004-2006

  This library implements a number of clustering algorithms:
  C-means, fuzzy C-means, and graph-based methods that basically 
  treat different connected components of certain graphs as clusters.

  Some of them work on the data matrix, assuming an underlying Euclidan metric.
  Some graph based algorithms rather work on a valued graph.
  

  */

#ifndef fff_CLUSTERING
#define fff_CLUSTERING

#ifdef __cplusplus
extern "C" {
#endif

#include "fff_graphlib.h"


 /*!
    \brief C-means algorithm
    \param Centers Cluster centers
    \param Label data labels
	\param X data matrix 
    \param maxiter maximum number of iterations
    \param delta small constant for control of convergence

    This algorithm performs a C-means clustering on the data X.
    Note that the data and cluster matrices should be dimensioned
    as (nb items * feature dimension) and (nb clusters * feature dimension)
    The metric used in the algorithm is Euclidian.
    The returned Label vector is a membership function 
    computed after convergence 
  */
  extern double fff_clustering_cmeans( fff_matrix* Centers, fff_array *Label, const fff_matrix* X,const  int maxiter,  double delta ); 
 /*!
    \brief Centroid computation in clustering methods
    \param Centers Cluster centers
    \param Label data labels
	\param X data matrix 

    This algorithm computes cluster centroids, given a certain labelling of the data
  */
  extern void fff_Estep( fff_matrix* Centers, const fff_array *Label, const fff_matrix* X);
/*!
    \brief Voronoi clustering of the dataset S, given Centers
    \param X data matrix 
    \param Centers Cluster centers
    \param Label data labels

    This algorithm performs the assignment step a C-means algorithm on the data X.
    Note that the data and cluster matrices should be dimensioned
    as (nb items * feature dimension) and (nb clusters * feature dimension)
    The metric used in the algo is Euclidian.
    The number of clusters is defined implicitly as Centers->size1
    Quick algo.
  */
  extern int fff_clustering_Voronoi ( fff_array *Label, const fff_matrix* Centers, const fff_matrix* X);
 /*!
    \brief Fuzzy C-means algorithm
    \param X data matrix 
    \param Centers Cluster centers
    \param Label data labels
    \param maxiter maximum number of iterations
    \param delta small constant for control of convergence

    This algorithm performs a Fuzzy C-means algorithm on the data X.
    The fuzziness index d is d=2.
    Note that the data and cluster matrices should be dimensioned
    as (nb items * feature dimension) and (nb clusters * feature dimension)
    The metric used in the algorithm is Euclidian.
    The returned Label vector is a hard membership function 
    computed after convergence 
  */
  extern int fff_clustering_fcm( fff_matrix* Centers, fff_array *Label, const fff_matrix* X, const int maxiter, const double delta ); 

   /*!
    \brief Labelling checking algorithm 
    \param Label vector
    \param k number of classes

    This algorithm quickly checks that for each i in [0,..,k-1]
    there exists an integer j so that Label[j] = i.
    it returns 1 if yes, 0 if no
  */
  extern int fff_clustering_OntoLabel(const fff_array * Label, const long k);

  /*
	\brief Ward clustering algorithm
	\param parent resulting tree-defining structure
	\param cost associated vector of merging cost
	\param X input data to be clustered
	
	This version is pretty efficient
   */
  extern int fff_clustering_ward(fff_array* parent, fff_vector *cost, const fff_matrix* X);


#ifdef __cplusplus
}
#endif

#endif
