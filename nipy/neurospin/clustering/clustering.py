from _clustering import *
from _clustering import __doc__

import numpy as np
from numpy.random import rand


def kmeans(X, nbclusters=2, Labels=None, maxiter=300, delta=0.0001,verbose=0):
    """ Centers, Labels, J = Cmeans(X,nbclusters,Labels,maxiter,delta)\n\
  cmeans clustering algorithm \n\
 INPUT :\n\
 - A data array X, supposed to be written as (n*p)\n\
   where n = number of features, p =number of dimensions\n\
 - nbclusters (int), the number of desired clusters\n\
 - Labels=None n array of predefined Labels. \n\
   if None or inadequate a random initilization is performed.\n\
 - maxiter(int, =300 by default), the maximum number \n\
   of iterations  before convergence\n\
 - delta(double, =0.0001 by default), \n\
  the relative increment in the results before declaring convergence\n\
  - verbose=0: verboseity mode\n\
 OUPUT :\n\
 - Centers: array of size nbclusters*p, the centroids of \n\
  the resulting clusters\n\
 - Labels : arroy of size n, the discrete labels of the input items;\n\
 - J the final value of the criterion
    """
    nbitems = X.shape[0]
    if nbitems<1:
        if verbose:
            raise ValueError, " I need at least one item to cluster"
        
    if np.size(X.shape)>2:
        if verbose:
            raise ValueError, " please enter a two-diemnsional array for clustering"
            
    if np.size(X.shape)==1:
        X = np.reshape(X,(nbitems,1))
    X = X.astype('d')
    
    nbclusters = int(nbclusters)
    if nbclusters<1:
        if verbose:
            print " cannot compute less than 1 cluster"
        nbclusters = 1
        
    if nbclusters>nbitems:
        if verbose:
            print " cannot find more clusters than items"
        nbclusters = nbitems
    
    nolabel = 1


    if Labels != None:
        if np.size(Labels) == nbitems:
            Labels = Labels.astype('i')
            OK = (Labels.min()>-1)&(Labels.max()<nbclusters+1)
            if OK:
                nolabel = 0
                maxiter = int(maxiter)
                if maxiter>0:
                    delta = float(delta)
                    if delta<0:
                        if verbose:
                            print "incorrect stopping criterion - ignored"
                        delta = 0.0001
                    else:
                        pass
                else:
                    if verbose:
                        print "incorrect number of iterations - ignored"
                    maxiter = 300
            else:
                if verbose:
                    print "incorrect labelling - ignored"
        else:
            if verbose:
                print "incompatible number of labels provided - ignored"
    
    if nolabel:
        Labels = (rand(nbitems)*nbclusters).astype('i')

    Centers,labels,J = cmeans(X, nbclusters, Labels, maxiter, delta)
         
    return Centers,labels,J
