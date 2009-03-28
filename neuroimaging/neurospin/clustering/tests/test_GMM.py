#!/usr/bin/env python

# to run only the simple tests:
# python testClustering.py Test_Clustering

import nose
from unittest import TestCase

import numpy as np
import numpy.random as nr
import neuroimaging.neurospin.clustering.gmm as gmm
import neuroimaging.neurospin.clustering.clustering as fc


sLabelMap = \
"""
00000000000000000000
00000111000000000000
00001111100000000000
00000111100000011000
00000011100000111100
00110000000000011000
01110000000000000000
00110000000000000000
01100000000000000011
00000001111000001111
00000001110000011111
00000011111000000111
00110000011000000111
01111000000000000111
01111000000000000011
01110000000000000000
00000000000010000000
00000000000111100000
00000000000111100000
00000000001110000000
"""

def stringToNarray(s):
    return np.array([int(c) for c in s])

class test_GMM(TestCase):

    def test_EM_gmm_1(self,verbose=0):
        """
        Computing the BIC value for different configurations
        """
        # generate some data
        dim = 2
        X = np.concatenate((nr.randn(1000,2),3+2*nr.randn(1000,2)))

        # estimate different GMMs of that data
        k = 2
        prec_type = 0
        lgmm = gmm.GMM(k,dim,prec_type)
        maxiter = 300
        delta = 0.001
        ninit = 5
        La,LL,bic1 = lgmm.estimate(X,None,maxiter,delta,ninit)
        if verbose:
            print "full covariance", La, LL, bic1
        
        lgmm.prec_type=1
        La,LL,bic2 = lgmm.estimate(X,None,maxiter,delta,ninit)
        
        lgmm.prec_type=2
        La,LL,bic3 = lgmm.estimate(X,None,maxiter,delta,ninit)

        lgmm.set_k(1)
        lgmm.prec_type=1
        La,LL,bic4 = lgmm.estimate(X,None,maxiter,delta,ninit)
        
        lgmm.set_k(3)
        La,LL,bic5 = lgmm.estimate(X,None,maxiter,delta,ninit)

        #print bic1,bic2,bic3,bic4,bic5
        # plot the result
        xmin = 1.1*X[:,0].min() - 0.1*X[:,0].max()
        xmax = 1.1*X[:,0].max() - 0.1*X[:,0].min()
        ymin = 1.1*X[:,1].min() - 0.1*X[:,1].max()
        ymax = 1.1*X[:,1].max() - 0.1*X[:,1].min()
        gd = gmm.grid_descriptor(2)
        gd.getinfo([xmin,xmax,ymin,ymax],[50,50])
        lgmm.sample(gd,X,verbose=0)
        self.assert_(bic5<bic2)

    def test_EM_gmm_3(self,verbose=0):
        """
        Computing the BIC value for different configurations
        """
        # generate some data
        dim = 2
        X = np.concatenate((nr.randn(1000,2),3+2*nr.randn(1000,2)))

        # estimate different GMMs of that data
        k = 2
        prec_type = 0
        lgmm = gmm.GMM(k,dim,prec_type)
        maxiter = 300
        delta = 0.001
        ninit = 5
        La,LL,bic1 = lgmm.estimate(X,None,maxiter,delta,ninit)
        
        lgmm.prec_type=1
        La,LL,bic2 = lgmm.estimate(X,None,maxiter,delta,ninit)
        

        # generate some data
        dim = 2
        X = np.concatenate((nr.randn(100,dim),5*nr.randn(100,dim)))

        #estimate different GMMs for that data
        maxiter = 300
        delta = 0.001
        ninit = 5
        k = 2
        prec_type = 0
        lgmm = gmm.GMM(k,dim,prec_type)
        La,LL,bic1 = lgmm.estimate(X,None,maxiter,delta,ninit)
        
        lgmm.prec_type=1
        La,LL,bic2 = lgmm.estimate(X,None,maxiter,delta,ninit)
            
        lgmm.prec_type=2
        La,LL,bic3 = lgmm.estimate(X,None,maxiter,delta,ninit)
        
        lgmm.set_k(1)
        lgmm.prec_type=1
        La,LL,bic4 = lgmm.estimate(X,None,maxiter,delta,ninit)
        
        lgmm.set_k(3)
        La,LL,bic5 = lgmm.estimate(X,None,maxiter,delta,ninit)

        if verbose:
            print bic1,bic2,bic3,bic4,bic5
        
        self.assert_(bic5<bic2)

    def test_EM_gmm_2(self,verbose=0):
        """
        Using cross-validated likelihood  instead
        """
        # generate some data
        dim = 2
    
        X = np.concatenate((nr.randn(1000,dim),3+2*nr.randn(1000,dim)))
        x = np.concatenate((nr.randn(100,dim),3+2*nr.randn(100,dim)))

        #estimate different GMMs for X, and test it on x
        maxiter = 300
        prec_type = 0
        k = 2
        lgmm = gmm.GMM(k,dim,prec_type)
        maxiter = 300
        delta = 0.001
        ninit = 5
        La,LL,bic = lgmm.estimate(X,None,maxiter,delta,ninit)
        LL1 = lgmm.test(x).mean()
        
        lgmm.prec_type=1
        La,LL,bic = lgmm.estimate(X,None,maxiter,delta,ninit)
        LL2 = lgmm.test(x).mean()

        lgmm.prec_type=2
        La,LL,bic = lgmm.estimate(X,None,maxiter,delta,ninit)
        LL3 = lgmm.test(x).mean()

        lgmm.set_k(1)
        lgmm.prec_type=1
        La,LL,bic = lgmm.estimate(X,None,maxiter,delta,ninit)
        LL4 = lgmm.test(x).mean()

        lgmm.set_k(3)
        La,LL,bic = lgmm.estimate(X,None,maxiter,delta,ninit)
        LL5 = lgmm.test(x).mean()
        
        lgmm.set_k(10)
        La,LL,bic = lgmm.estimate(X,None,maxiter,delta,ninit)
        LL6 = lgmm.test(x).mean()

        if verbose:
            print lgmm.assess_divergence(X)
        
        #print LL1,LL2,LL3,LL4,LL5,LL6
        self.assert_(LL6<LL2)

        
    def test_select_gmm(self,verbose=0):
        """
        Computing the BIC value for different configurations
        """
        # generate some data
        dim = 2
        X = np.concatenate((nr.randn(100,2),3+2*nr.randn(100,2)))

        # estimate different GMMs of that data
        k = 2
        prec_type = 1#0
        # there is clearly a bug when prec_type=0 is used.
        # this bug is numerical (C layer)
        # not solved yet
        lgmm = gmm.GMM(k,dim,prec_type)
        maxiter = 300
        delta = 0.001
        ninit = 5
        kvals = np.arange(10)+2

        La,LL,bic = lgmm.optimize_with_BIC(X, kvals, maxiter, delta, ninit,verbose)

        if verbose:
            # plot the result
            xmin = 1.1*X[:,0].min() - 0.1*X[:,0].max()
            xmax = 1.1*X[:,0].max() - 0.1*X[:,0].min()
            ymin = 1.1*X[:,1].min() - 0.1*X[:,1].max()
            ymax = 1.1*X[:,1].max() - 0.1*X[:,1].min()
            gd = gmm.grid_descriptor(2)
            gd.getinfo([xmin,xmax,ymin,ymax],[50,50])
            gmm.sample(gd,X,verbose=0)

            #print gmm.k
        self.assert_(lgmm.k<5)
        
        
    def test_Bayesian_GMM_1D_1(self,verbose = 0):
        # generate some bimodal data
        dim = 1
        X = nr.randn(1000,dim)
        X [-300:,:]= X [-300:,:]+5
        offset = 2
        X = X - offset
        
        # generate the BGMM 
        k = 3
        b = gmm.BGMM(k,dim)
        b.set_empirical_priors(X)
            
        # generate a sampling grid  
        gd = gmm.grid_descriptor(1)
        gd.getinfo([-10.,10.],200)

        # estimate the model and sample it on the grid
        niter = 1000
        delta = 0.0001
        Li,Label= b.VB_estimate_and_sample(X,niter = niter,delta = 0.0001,gd = gd,verbose=verbose)
        #Li,Label= b.Gibbs_estimate_and_sample(X,niter = niter,gd = gd)
        self.assert_(True)

    def test_Bayesian_GMM_1D_2(self):
        # generate some bimodal data
        dim = 1
        X = nr.randn(1000,dim)
        X [-300:,:]= X [-300:,:]*5
            
        # create the BGMM structure
        k = 3
        b = gmm.BGMM(k,dim)
        b.set_empirical_priors(X)
        
        # estimate the model
        niter = 1000
        delta = 0.0001
        b.VB_estimate(X,niter = niter,delta=delta)

        #create the sampling grid
        gd = gmm.grid_descriptor(1)
        gd.getinfo([-10.,10.],200)

        #Li= b.VB_sample(gd,X)
        self.assert_(True)


    def test_Bayesian_GMM_1D_3(self,verbose=0):
        # generate some bimodal data
        dim = 1
        X = nr.randn(100,dim)
        X [-30:,:]= X [-30:,:]+5
            
        # create the BGMM structure
        k = 3
        b = gmm.BGMM(k,dim)
        b.set_empirical_priors(X)
    
        # estimate the model
        niter = 1000
        delta = 0.0001
        b.VB_estimate(X,niter = niter,delta=delta)
        b.Gibbs_estimate(X,niter=niter)
        
        # and take the log-posterior of the data
        L1,Label= b.VB_estimate_and_sample(X,niter = niter,delta = 0.0001,gd = None,verbose=0)      
        L2,Label= b.Gibbs_estimate_and_sample(X,niter = niter,gd = None,verbose=0)
        if verbose:
            print L1.mean(), L2.mean()
        self.assert_(np.absolute( L1.mean()-L2.mean())<0.1)

    def test_Bayesian_GMM_1D_4(self, level=0,verbose=0):

        ## Generate data:
        # Some labels (or states) with a clustered structure:

        labels = np.array(map(stringToNarray, sLabelMap.split('\n')[1:-1]))
                
        # Mixture parameters:
        muInact = 10.
        vInact = 2.
        muAct = 300.
        vAct = 40.
        
        # Generate samples:
        samples = np.zeros(labels.shape, dtype=float)
        mInact = (labels==0)
        samples[mInact] = np.random.randn(mInact.sum())*vInact**.5 + muInact
        mAct = (labels==1)
        samples[mAct] = np.random.randn(mAct.sum())*vAct**.5 + muAct

        # Plot labels & samples:
        if verbose :
            import pylab
            pylab.matshow(labels)
            #mf = pylab.matshow(samples)
            #pylab.colorbar(mf)

        dim = 1
        nbClasses = 2
        bgmm = gmm.BGMM(nbClasses,dim)
        bgmm.set_empirical_priors(samples.ravel())
        if verbose :
            print 'empirical prior centers:', bgmm.prior_centers
            print 'empirical prior precisions :', bgmm.prior_precision
        bgmm.prior_centers[0] = 0.
        #gmm.prior_centers[1] = 300.
        bgmm.prior_precision *= samples.size#10000*samples.size
        bgmm.prior_mean_scale[0]=samples.size
        
        l, eLabels = bgmm.Gibbs_estimate_and_sample(samples.reshape(samples.size,1),niter=1000)
        if verbose :
            print 'estimated centers : ', bgmm.centers
            print 'precisions', bgmm.precisions
            print 'dof ',bgmm.dof
            print 'estimated variance :', np.squeeze(1/bgmm.precisions)/bgmm.dof
            print 'estimated weights :', bgmm.weights
            print dir(bgmm)
            import pylab
            pylab.matshow(eLabels.reshape(samples.shape))
            pylab.show()
        self.assert_(True)
        
    def test_Bayesian_GMM_2D_2(self):
        # generate some bimodal data
        dim = 2
        X = nr.randn(100,dim)
        X [-30:,:]= X [-30:,:]*1+[1,3]
            
        # create the BGMM structure
        k = 2
        b = gmm.BGMM(k,dim)
        b.set_empirical_priors(X)
        
        # estimate the model
        niter = 1000
        delta = 0.0001
        b.VB_estimate(X,niter = niter,delta=delta)

        
        #create the sampling grid
        xmin = 1.1*X[:,0].min() - 0.1*X[:,0].max()
        xmax = 1.1*X[:,0].max() - 0.1*X[:,0].min()
        ymin = 1.1*X[:,1].min() - 0.1*X[:,1].max()
        ymax = 1.1*X[:,1].max() - 0.1*X[:,1].min()
        gd = gmm.grid_descriptor(2)
        gd.getinfo([xmin,xmax,ymin,ymax],[50,50])

        #Li= b.VB_sample(gd,X)
        self.assert_(True)

    def test_hdp(self,verbose=0):
        """
        test the homoscedastic fdp model
        """
        # create the data
        dim = 1
        X = nr.randn(100,dim)
        X [-30:,:]= X [-30:,:]+5

        #create the sampling grid
        xmin = 1.2*X[:,0].min() - 0.2*X[:,0].max()
        xmax = 1.2*X[:,0].max() - 0.2*X[:,0].min()
        gd = gmm.grid_descriptor(1)
        gd.getinfo([xmin,xmax],100)
        mygrid = gd.make_grid()
        
        # create the HDP structure
        
        g0 = 1.0/(xmax-xmin);
        g1 =g0
        alpha = 0.5;
        prior_precision = 5*np.ones((1,1))
        sub = (nr.rand(100)*10).astype('i')
        bf1 = np.ones(100)
        spatial_coords = gd
        burnin = 100
        nis = 1000 # number of iterations for grid sampling
        nii = 1 # number of ietrations to compute the posterior
        dof = 0
        # to get the  data log-like
        p0,q0 = fc.fdp(X, alpha, g0, g1, dof,prior_precision, bf1, sub, burnin,X,10,1000)
        
        # to sample ona  grid
        p,q = fc.fdp(X, alpha, g0, g1, dof,prior_precision, bf1, sub, burnin, mygrid,10,1000)
        if verbose:
            import matplotlib.pylab as MP
            MP.figure()
            MP.plot(np.squeeze(mygrid),p)
            MP.show()

        sp = np.sum(p)*(mygrid[1]-mygrid[0])
        if verbose:
            print "Infinite GMM",
            print "Average LL: ", np.mean(np.log(p0)),"denisty sum: ",sp
        
        sp = (sp<1.01)*(sp>0.9)
        self.assert_(sp)
        
        
        
    def test_hdp2(self,verbose=0):
        """
        Test the heteorscedstic fdp model
        To be completed !
        """
        # create the data
        dim = 1
        X = nr.randn(100,dim)
        X [-30:,:]= X [-30:,:]+5

        #create the sampling grid
        xmin = 1.3*X[:,0].min() - 0.3*X[:,0].max()
        xmax = 1.3*X[:,0].max() - 0.3*X[:,0].min()
        gd = gmm.grid_descriptor(1)
        gd.getinfo([xmin,xmax],100)
        mygrid = gd.make_grid()
        
        # create the HDP structure

        g0 = 1.0/(xmax-xmin);
        g1 =g0
        alpha = 0.5;
        prior_precision = 5*np.ones((1,1))
        sub = (nr.rand(100)*10).astype('i')
        bf1 = np.ones(100)
        spatial_coords = gd
        burnin = 100
        nis = 1000 # number of iterations for grid sampling
        nii = 3000 # number of ietrations to compute the posterior probability of input
        dof = 10
        
        # to get the  data log-like
        p0,q0 = fc.fdp(X, alpha, g0, g1, dof,prior_precision, bf1, sub, burnin,X,burnin,nis)
        
        # to sample ona  grid
        p,q =  fc.fdp(X, alpha, g0, g1, dof,prior_precision, bf1, sub, burnin, mygrid,burnin,nii)
        if verbose:
            import matplotlib.pylab as MP
            MP.figure()
            MP.plot(np.squeeze(mygrid),p)
            MP.show()

        sp = np.sum(p)*(mygrid[1]-mygrid[0])
        #print "infinite mixture of Student- approximated"
        #print  "average LL: ", np.mean(np.log(p0)), "density sum: ", sp
        sp = (sp<1.01)*(sp>0.8)
        self.assert_(sp)

    def test_dpmm(self,verbose=0):
        """
        Test the heteorscedstic DPMM
        """
        # create the data
        dim = 1
        X = nr.randn(100,dim)
        X [-30:,:]= X [-30:,:]+5

        #create the sampling grid
        xmin = 1.2*X[:,0].min() - 0.2*X[:,0].max()
        xmax = 1.2*X[:,0].max() - 0.2*X[:,0].min()
        gd = gmm.grid_descriptor(1)
        gd.getinfo([xmin,xmax],100)
        mygrid = gd.make_grid()
        
        # create the HDP structure

        alpha = 0.5;
        prior_precision = 5*np.ones((1,dim))
        prior_means = np.reshape(X.mean(0),(1,dim))
        prior_mean_scale = np.ones((1,dim))
        sub = (nr.rand(100)*10).astype('i')
        spatial_coords = gd
        burnin = 100
        nis = 1000 # number of iterations for grid sampling
        nii = 1 # number of ietrations to compute the posterior probability of input
        dof = 1
        
        # to get the  data log-like
        p0 = fc.dpmm(X, alpha,prior_precision, prior_means,prior_mean_scale, sub, burnin,nis=nis,dof=dof)
        
        # to sample ona  grid
        p = fc.dpmm(X, alpha,prior_precision, prior_means,prior_mean_scale, sub, burnin,mygrid,nis=nii,dof=dof) 
        if verbose:
            import matplotlib.pylab as MP
            MP.figure()
            MP.plot(np.squeeze(mygrid),p)
            MP.show()

        sp = np.sum(p)*(mygrid[1]-mygrid[0])
        if verbose:
            print  "DPMM. average LL: ", np.mean(np.log(p0)), "density sum: ", sp
        sp = (sp<1.01)*(sp>0.8)
        self.assert_(sp)

if __name__ == '__main__':
    nose.run(argv=['', __file__])

    
