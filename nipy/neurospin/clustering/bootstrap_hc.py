"""
This module provides some code to perform bootstrap
of Ward's hierarchical clustering
This is useful to statistically validate clustering results.
theory see: 

Author : Bertrand Thirion, 2008
"""

#---------------------------------------------------------------------------
# ------ Routines for Agglomerative Hierarchical Clustering ----------------
# --------------------------------------------------------------------------

import numpy as np
from nipy.neurospin.clustering.hierarchical_clustering import Ward_simple
from numpy.random import random_integers

# -------------------------------------------------------------------
# ---- bootstrap procedure for Ward algorithm -----------------------
# -------------------------------------------------------------------

def _compare_list_of_arrays(l1, l2):
    """
    INPUT:
        l1 and l2 are two lists of 1D arrays.
    OUTPUT:
        An 1D array 'OK' of same shape as l1, with:
        - OK[i] if there is a element l of l2 such as l1[i] is a
            permutation of l.
        - 0 elsewhere.
    """
    OK = np.zeros(len(l1), 'i')
    # Sort the arrays in l1 and l2
    l1 = [np.sort(l) for l in l1]
    l2 = [np.sort(l) for l in l2]
    for index, col1 in enumerate(l1):
        for col2 in l2:
            if np.all(col1 == col2):
                OK[index] = 1
                break

    return OK   


def _bootstrap_cols(x, p=-1):
    """
    create a colomn_wise bootstrap of x
    INPUT:
    - x an (m,n) array
    - p=-1 the number of serires rows. By default, p=n
    OUPUT
    - y an(m,p) array such that y[:,i] is a column of x for each i
    """
    _, n = x.shape
    if p==-1:
        p = n
    indices = random_integers(0, n-1, size=(p, ))
    y = x[:,indices]
    return y


def ward_msb(X, niter=1000):
    """
    multi-scale bootstrap procedure
    INPUT:
    - X array of shape (n,p) where
    n is the number of items to be clustered
    p is their dimensions
    - niter=1000
    number of iterations of the bootstrap
    OUPUT:
    - t the resulting tree clustering
    the associated subtrees is defined as t.list_of_subtrees()
    there are precisely n such subtrees
    - cpval: array of shape (n) : the corrected p-value of the clusters
    - upval: array of shape (n) : the uncorrected p-value of the clusters
    """
    from scipy.special import erf,erfinv
    from numpy.linalg import pinv

    n = X.shape[0]
    d = X.shape[1]
    t = Ward_simple(X)
    l = t.list_of_subtrees()

    db = (d*np.exp((np.arange(7)-3)*np.log(1.1))).astype('i')
    pval = np.zeros((len(l),len(db)))

    # get the bootstrap samples
    for j in range(len(db)):
        for i in range(niter):
            # nb : spends 95% of the time in ward algo
            # when n=100,d=30
            x = _bootstrap_cols(X,db[j])
            t = Ward_simple(x)
            laux = t.list_of_subtrees()
            pval[:,j] += _compare_list_of_arrays(l, laux)

    # collect the empirical pval for different boostrap sizes
    pval = pval/niter
    upval = pval[:,3]

    # apply the correction
    tau = np.sqrt(float(d)/db)
    u = np.vstack((tau,1.0/tau))
    z = -np.sqrt(2)*erfinv(2*pval-1)
    r = np.dot(z,pinv(u))
    zc = r[:,0]-r[:,1]
    cpval = 0.5*(1+erf(zc/np.sqrt(2)))
    
    # when upval is too small, force cpval to 0
    cpval[upval<1.0/n]=0

    return t,cpval,upval

def demo_ward_msb(n=30, d=30, niter=1000):
    """
    basic demo for the ward_msb procedure
    in that case the dominant split with 2 clusters should have
    dominant p-val
    INPUT:
    - n,d : the dimensions of the dataset
    -niter : the number of bootrstraps
    """
    from numpy.random import randn
    X = randn(n,d)
    X[:np.ceil(n/3)] += 1.0
    niter = 1000
    t, cpval, upval = ward_msb(X, niter)
    t.plot()
    import matplotlib.pylab as MP
    MP.figure()
    MP.plot(upval,'o')
    MP.plot(cpval,'o')
    MP.show()

