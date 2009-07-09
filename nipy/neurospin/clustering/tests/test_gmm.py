#!/usr/bin/env python

# to run only the simple tests:
# python testClustering.py Test_Clustering

import numpy as np
import numpy.random as nr
from nipy.neurospin.clustering.gmm import GMM
import nipy.neurospin.clustering.gmm as gmm
import nipy.neurospin.clustering.clustering as fc



def test_em_selection():
    """
    test that the basic GMM-based model selection tool
    returns something sensible
    (i.e. the gmm used to represent the data has indeed one or two classes)
    """
    # generate some data
    dim = 2
    x = np.concatenate((nr.randn(100,dim),3+2*nr.randn(100,dim)))

    krange = range(1,10)
    lgmm = gmm.best_fitting_GMM(x,krange,prec_type='full',
                                niter=100,delta = 1.e-4,ninit=1,verbose=0)
    assert (lgmm.k<4)
    

def test_em_gmm_full(verbose=0):
    """
    Computing the BIC value for different configurations
    of a GMM with ful diagonal matrices
    The BIC should be maximal for a number of classes of 1  or 2
    """
    # generate some data
    dim = 2
    x = np.concatenate((nr.randn(100,dim),3+2*nr.randn(100,dim)))
    
    # estimate different GMMs of that data
    maxiter = 100
    delta = 1.e-4

    bic = np.zeros(5)
    for k in range(1,6):
        lgmm = GMM(k,dim)
        lgmm.initialize(x)
        bic[k-1] = lgmm.estimate(x,maxiter,delta,verbose)
        if verbose: print "bic of the %d-classes model"%k, bic

    z = lgmm.map_label(x)
    assert(bic[4]<bic[1])


def test_em_gmm_diag(verbose=0):
    """
    Computing the BIC value for GMMs with different number of classes,
    with diagonal covariance models
    The BIC should maximal for a number of classes of 1  or 2
    """
    # generate some data
    dim = 2
    x = np.concatenate((nr.randn(1000,dim),3+2*nr.randn(1000,dim)))
    
    # estimate different GMMs of that data
    maxiter = 100
    delta = 1.e-8
    prec_type='diag'

    bic = np.zeros(5)
    for k in range(1,6):
        lgmm = GMM(k,dim,prec_type)
        lgmm.initialize(x)
        bic[k-1] = lgmm.estimate(x,maxiter,delta,verbose)
        if verbose: print "bic of the %d-classes model"%k, bic

    z = lgmm.map_label(x)

    assert((z.max()+1==lgmm.k)&(bic[4]<bic[1]))

def test_em_gmm_multi(verbose=0):
    """
    Playing with various initilizations on the same data
    """
    # generate some data
    dim = 2
    x = np.concatenate((nr.randn(1000,dim),3+2*nr.randn(100,dim)))
    
    # estimate different GMMs of that data
    maxiter = 100
    delta = 1.e-4
    ninit = 5
    k = 2
    
    lgmm = GMM(k,dim)
    bgmm = lgmm.initialize_and_estimate(x,maxiter,delta,ninit,verbose)
    bic = bgmm.evidence(x)
    
    if verbose: print "bic of the best model", bic

    if verbose:
        # plot the result
        from test_bgmm import plot2D
        z = lgmm.map_label(x)
        plot2D(x,lgmm,z,show = 1,verbose=0)

    assert (np.isfinite(bic))
    
def test_em_gmm_largedim(verbose=0):
    """
    testing the GMM model in larger dimensions
    """
    # generate some data
    dim = 10
    x = nr.randn(100,dim)
    x[:30,:] +=1
    
    # estimate different GMMs of that data
    maxiter = 100
    delta = 1.e-4
    
    for k in range(1,3):
        lgmm = GMM(k,dim)
        lgmm.initialize(x)
        bic = lgmm.estimate(x,maxiter,delta,verbose)
        if verbose: print "bic of the %d-classes model"%k, bic
        
    z = lgmm.map_label(x)
    
    # define the correct labelling
    u = np.zeros(100)
    u[:30]=1

    #check the correlation between the true labelling
    # and the computed one
    eta = np.absolute(np.dot(z-z.mean(),u-u.mean())/(np.std(z)*np.std(u)*100))
    assert (eta>0.3)

def test_em_gmm_heterosc(verbose=0):
    """
    testing the model in very ellipsoidal data:
    compute the big values for several values of k
    and check that the macimal is 1 or 2
    """
    # generate some data
    dim = 2
    x = nr.randn(100,dim)
    x[:50,:] +=3
    #x[:,0]*=10
    
    # estimate different GMMs of that data
    maxiter = 100
    delta = 1.e-4
    
    bic = np.zeros(5)
    for k in range(1,6):
        lgmm = GMM(k,dim)
        lgmm.initialize(x)
        bic[k-1] = lgmm.estimate(x,maxiter,delta,0)
        if verbose: print "bic of the %d-classes model"%k, bic

    if verbose:
        # plot the result
        z = lgmm.map_label(x)
        from test_bgmm import plot2D
        plot2D(x,lgmm,z,show = 1,verbose=0)
    assert (bic[4]<bic[1])

        
def test_em_gmm_cv(verbose=0):
    """
    Comparison of different GMMs using cross-validation
    """
    # generate some data
    dim = 2
    xtrain = np.concatenate((nr.randn(100,dim),3+2*nr.randn(100,dim)))
    xtest = np.concatenate((nr.randn(1000,dim),3+2*nr.randn(1000,dim)))
    
    #estimate different GMMs for xtrain, and test it on xtest
    prec_type = 'full'
    k = 2        
    maxiter = 300
    delta = 1.e-4
    ll = []
    
    # model 1
    lgmm = GMM(k,dim,prec_type)
    lgmm.initialize(xtrain)
    bic = lgmm.estimate(xtrain,maxiter,delta)
    ll.append(lgmm.test(xtest).mean())
    
    prec_type='diag'
    # model 2
    lgmm = GMM(k,dim,prec_type)
    lgmm.initialize(xtrain)
    bic = lgmm.estimate(xtrain,maxiter,delta)
    ll.append(lgmm.test(xtest).mean())
        
    for  k in [1,3,10]:
        lgmm = GMM(k,dim,prec_type)
        lgmm.initialize(xtrain)
        bic = lgmm.estimate(xtrain,maxiter,delta)
        ll.append(lgmm.test(xtest).mean())
            
    assert(ll[4]<ll[1])

def test_select_gmm_old_full(verbose=0):
    """
    Computing the BIC value for different configurations
    """
    # generate some data
    dim = 2
    x = np.concatenate((nr.randn(100,2),3+2*nr.randn(100,2)))
    
    # estimate different GMMs of that data
    k = 2
    prec_type = 'full'

    lgmm = gmm.GMM_old(k,dim,prec_type)
    maxiter = 300
    delta = 0.001
    ninit = 5
    kvals = np.arange(10)+2
    
    La,LL,bic = lgmm.optimize_with_bic(x, kvals, maxiter, delta, ninit,verbose)

    if verbose:
        # plot the result
        xmin = 1.1*X[:,0].min() - 0.1*X[:,0].max()
        xmax = 1.1*X[:,0].max() - 0.1*X[:,0].min()
        ymin = 1.1*X[:,1].min() - 0.1*X[:,1].max()
        ymax = 1.1*X[:,1].max() - 0.1*X[:,1].min()
        gd = gmm.grid_descriptor(2)
        gd.getinfo([xmin,xmax,ymin,ymax],[50,50])
        gmm.sample(gd,X,verbose=0)
        
    assert(lgmm.k<5)
    
def test_select_gmm_old_diag(verbose=0):
    """
    Computing the BIC value for different configurations
    """
    # generate some data
    dim = 2
    x = np.concatenate((nr.randn(100,2),3+2*nr.randn(100,2)))
    
    # estimate different GMMs of that data
    k = 2
    prec_type = 'diag'

    lgmm = gmm.GMM_old(k,dim,prec_type)
    maxiter = 300
    delta = 0.001
    ninit = 5
    kvals = np.arange(10)+2
    
    La,LL,bic = lgmm.optimize_with_bic(x, kvals, maxiter, delta, ninit,verbose)

    if verbose:
        # plot the result
        xmin = 1.1*X[:,0].min() - 0.1*X[:,0].max()
        xmax = 1.1*X[:,0].max() - 0.1*X[:,0].min()
        ymin = 1.1*X[:,1].min() - 0.1*X[:,1].max()
        ymax = 1.1*X[:,1].max() - 0.1*X[:,1].min()
        gd = gmm.grid_descriptor(2)
        gd.getinfo([xmin,xmax,ymin,ymax],[50,50])
        gmm.sample(gd,X,verbose=0)
        
    assert(lgmm.k<5)


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])


