from _clustering import *
from _clustering import __doc__

import numpy as np
from numpy.random import rand


def kmeans(X, nbclusters=2, Labels=None, maxiter=300, delta=0.0001,verbose=0):
    """
    kmeans clustering algorithm 

    Parameters
    ----------
    X: array of shape (n,p): n = number of items, p = dimension
       data array
    nbclusters (int), the number of desired clusters
    Labels = None array of shape (n) prior Labels.   
           if None or inadequate a random initilization is performed.
    maxiter=300 (int), the maximum number of iterations  before convergence
    delta=0.0001 (float), 
                 the relative increment in the results 
                 before declaring convergence.
    verbose=0: verboseity mode
 
    Returns
    -------
    Centers: array of size nbclusters*p, 
             the centroids of  the resulting clusters
    Labels : array of size n, the discrete labels of the input items
    J (float):  the final value of the inertia criterion
    """
    nbitems = X.shape[0]
    if nbitems<1:
        if verbose:
            raise ValueError, " I need at least one item to cluster"
        
    if np.size(X.shape)>2:
        if verbose:
            raise ValueError, " please enter a two-diemnsional array \
                              for clustering"
            
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
            Labels = Labels.astype(np.int)
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
        Labels = (rand(nbitems)*nbclusters).astype(np.int)

    Centers,labels,J = cmeans(X, nbclusters, Labels, maxiter, delta)
         
    return Centers,labels,J
