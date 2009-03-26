#!/usr/bin/env python

# to run only the simple tests:
# python testClustering.py Test_Clustering

import neuroimaging.neurospin.clustering.clustering as fc
import nose
import numpy as np
import numpy.random as nr
from unittest import TestCase


class TestClustering(TestCase):

    def testcmeans1(self):
        X = nr.randn(10,2)
        A = np.concatenate([np.ones((7,2)),np.zeros((3,2))])
        X = X+3*A;
        C,L,J = fc.cmeans(X,2)
        L = np.array([0,0,0,0,0,1,1,1,1,1])
        C,L,J = fc.cmeans(X,2,L)
        self.assert_(np.mean(L[:7])<0.5)

    def testcmeans2(self):
        X = nr.randn(10000,2)
        A = np.concatenate([np.ones((7000,2)),np.zeros((3000,2))])
        X = X+3*A
        L = np.concatenate([np.ones(5000), np.zeros(5000)]).astype('i')
        C,L,J = fc.cmeans(X,2,L)
        l = L[:7000].astype('d')
        self.assert_(np.mean(l)>0.9)


    def testvoronoi(self):
        X = nr.randn(10000,2)
        A = np.concatenate([np.ones((7000,2)),np.zeros((3000,2))])
        X = X+3*A
        C = np.array([[0,0],[3,3]])
        L = fc.voronoi(X,C)
        l = L[:7000].astype('d')
        self.assert_(np.mean(l,0)>0.5)

    def testfcm(self):
        X = nr.randn(10,2)
        A = np.concatenate([np.ones((7,2)),np.zeros((3,2))])
        X = X+3*A
        #raise Exception, """Test failing and corrupting later tests.
        #FCM is not working. Temporarily skipping this test."""
        C,L = fc.fcm(X,2)
        #print C
        C,L,J = fc.cmeans(X,2,L)
        #print C
        self.assert_(True)

class TestGMM(TestCase):
    
    def testgmm1(self):
        X = nr.randn(10000,2)
        A = np.concatenate([np.ones((7000,2)),np.zeros((3000,2))])
        X = X+3*A
        L = np.concatenate([np.ones(5000), np.zeros(5000)]).astype('i')
        C,P,W,L,J = fc.gmm(X,2,L);
        l = L[:7000].astype('d') 
        self.assert_(np.mean(l)>0.9)

    def testgmm1_andCheckResult(self):
        np.random.seed(0) # force the random sequence
        X = nr.randn(10000,2)
        A = np.concatenate([np.ones((7000,2)),np.zeros((3000,2))])
        X = X+3*A
        L = np.concatenate([np.ones(5000), np.zeros(5000)]).astype('i')
        C,P,W,L,J = fc.gmm(X,2,L); 
        np.random.seed(None) # re-randomize the seed
        # results for randomseed = 0
        expectC = np.concatenate([np.zeros((1,2)),3*np.ones((1,2))])
        expectP = np.ones((2,2))
        expectW = np.array([ 0.3, 0.7])
        expectSL = 7000;
        self.assert_( np.allclose(C, expectC,0.01,0.05 ))
        self.assert_( np.allclose(P, expectP,0.03,0.05))
        self.assert_( np.allclose(W, expectW ,0.03,0.05))
        self.assert_( np.allclose(expectSL, L.sum(),0.01))

    def testgmm2(self):
        X = nr.randn(10000,2)
        A = np.concatenate([np.ones((7000,2)),np.zeros((3000,2))])
        X = X+3*A
        C,P,W,L,J = fc.gmm(X,2)
        dW= W[0]-W[1]
        ndW = dW*dW
        self.assert_(ndW > 0.1)
      
    def testgmm3(self):
        X = nr.randn(10000,2)
        A = np.concatenate([np.ones((7000,2)),np.zeros((3000,2))])
        X = X+3*A
        L = np.concatenate([np.ones(5000), np.zeros(5000)]).astype('i')
        C,P,W,L,J = fc.gmm(X,2,L,2)
        C,P,W,L,J = fc.gmm(X,2,L,1)
        C,P,W,L,J = fc.gmm(X,2,L,0)
        l = L[:7000].astype('d')
        self.assert_(np.mean(l)>0.9)  

    def testpartition(self):
        X = nr.randn(10000,2)
        A = np.concatenate([np.ones((7000,2)),np.zeros((3000,2))])
        X = X+3*A
        C = np.array([[0,0],[3,3]])
        P = np.array([[1,1],[1,1]])
        W = np.array([0.5, 0.5])
        L,G = fc.gmm_partition(X,C,P,W)
        l = L[:7000].astype('d')
        self.assert_(np.mean(l)>0.5)


    def test_Gibbs_GMM(self, verbose=0):
        k = 2
        dim = 2
        # prior_means = np.zeros((k,dim),'d')
        prior_means = np.concatenate([np.zeros((1,dim)),np.ones((1,dim))])
        prior_precision_scale =  1*np.ones((k,dim),'d') # 0.01
        prior_mean_scale = 1*np.ones(k,'d')
        prior_weights = np.ones(k,'d')
        prior_dof =  (dim+1)*np.ones(k,'d') # 100
        X = nr.randn(100,dim)-1
        X[-30:] = X[-30:]+ 4
        membership, mean, mean_scale,precision_scale, weights,dof,density = fc.gibbs_gmm(X, prior_means,prior_precision_scale,prior_mean_scale,prior_weights,prior_dof,1000)
        expectC = np.array([[-1,-1],[3,3]])
        if verbose:
            print expectC,mean
        self.assert_( np.allclose(expectC, mean,0.3,0.3))


class TestTypeProof(TestCase):

    def testtemplate(self):
        X = nr.randn(10,2)
        A = np.vstack(( np.ones((7,2)), np.zeros((3,2)) ))
        X = X + 3*A
        C,L,J = fc.cmeans(X,2)
        L = np.array([0,0,0,0,0,1,1,1,1,1])
        C,L,J = fc.cmeans(X,2,L)
        self.assert_(np.mean(L[:7])<0.5)

    # basic typecheck or argcheck
    def testarg1(self):
        X = nr.randn(10,2)
        A = np.vstack(( np.ones((7,2)), np.zeros((3,2)) ))
        X = X + 3*A
        C,L,J = fc.kmeans(X,2.0)
        C,L,J = fc.kmeans(X,0.5)
        C,L,J = fc.kmeans(X,-42)
        self.assert_(True)

    def testarg2(self):
        X = nr.randn(10,2)
        A = np.vstack(( np.ones((7,2)), np.zeros((3,2)) ))
        L = np.array([0,0,0,0,0,1,1,1,1,1])
        C,L,J = fc.kmeans(X,2,L)
        L = np.array([0.0,0,0,0,0,1,1,1,1,1.42])
        C,L,J = fc.kmeans(X,2,L)
        L = np.array([0.0,0,0,0,0,1,1,1,1,-1])
        C,L,J = fc.kmeans(X,2,L)
        self.assert_(True)

    def testarg3(self):
        A = np.vstack(( np.ones((7,2)), np.zeros((3,2)) ))
        X = (nr.randn(10,2) * 100).astype('i')
        C,L,J = fc.kmeans(X,2)
        C,L,J = fc.kmeans(X,2)
        X = nr.randn(1,2)
        C,L,J = fc.kmeans(X,2)
        X = nr.randn(1,2)
        C,L,J = fc.kmeans(X,2)
        X = nr.randn(2,5)
        C,L,J = fc.kmeans(X,30)
        self.assert_(True)

    def testvarious(self):
        import sys
        import weakref
        Y = nr.randn(20,2)
        X = Y[::2, :]
        del Y
        self.assert_(X.flags['CONTIGUOUS'] == False)

        A = np.vstack(( np.ones((7,2)), np.zeros((3,2)) ))
        X = X + 3*A
        wX = weakref.ref(X)
        self.assert_(sys.getrefcount(X) == 2)

        L1 = np.array([0,0,0,0,0,1,1,1,1,1])
        C,L,J = fc.cmeans(X,2,L1)
        self.assert_(id(L1) != id(L))
        
        C,L,J = fc.cmeans(X,2,L)
        del X
        self.assert_(wX() == None)


    #def testmmap(self):
    #   Y = np.memmap(join(split(__file__)[0],'data','raw_double_array_bigendian'), float, 'r', shape=(50, 10))
    #   X = Y[20:40, 2:5]
    #   self.assert_(np.shape(X) == (20, 3))
    #
    #   C,L,J = fc.cmeans(X,5)
    #   self.assert_(True)


if __name__ == '__main__':
    nose.run(argv=['', __file__])
