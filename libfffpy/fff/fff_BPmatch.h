/*!
  \file fff_BPmatch.h
  \brief Discrete matching function based on BP networks
  \author Bertrand Thirion
  \date 2006

  This library implements a discrete matching function, 
  which is based on a belief propagation algorithm. 
  It is not intended to remain as such in fff, 
  but will be embedded in a more systematic framework.
  
  */

#ifndef fff_BPMATCH
#define fff_BPMATCH

#ifdef __cplusplus
extern "C" {
#endif

#include "fff_graphlib.h"
#include "fff_array.h"
#include "fff_vector.h"
#include "fff_matrix.h"
#include "fff_base.h"
#include "fff_blas.h"


  /*!
  \brief Discrete matching algorithm
  \param source matrix of the source data
  \param target matrix of the source data
  \param adjacency matrix of the graph related to X
  \param belief matrix of the resulting matching belifs of the elements of X
  \param d0  distance considered in the matching 
  */
  
  extern int fff_BPmatch(fff_matrix * source, fff_matrix * target, fff_matrix * adjacency, fff_matrix * belief, double d0);
 

#ifdef __cplusplus
}
#endif

#endif
