# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import unittest

import numpy as np

from ..displacement_field import displacement_field, gaussian_random_field

def make_data(n=10, dim=20, r=5, mdim=15, maskdim=20, amplitude=10, noise=1, jitter=None, activation=False):
    XYZvol = np.zeros((dim,dim,dim),int)
    XYZ = np.array(np.where(XYZvol==0))
    p = XYZ.shape[1]
    #mask = np.arange(p)
    XYZvol[XYZ[0],XYZ[1],XYZ[2]] = np.arange(p)
    o = np.array([dim/2,dim/2,dim/2])
    I = XYZvol[(dim-mdim)/2:(dim+mdim)/2,(dim-mdim)/2:(dim+mdim)/2,(dim-mdim)/2:(dim+mdim)/2].ravel()
    mask = XYZvol[ (dim-maskdim)/2 : (dim+maskdim)/2, (dim-maskdim)/2 : (dim+maskdim)/2, (dim-maskdim)/2 : (dim+maskdim)/2 ].ravel()
    q = len(mask)
    maskvol = np.zeros((dim,dim,dim),int)
    maskvol[XYZ[0,mask],XYZ[1,mask],XYZ[2,mask]] = np.arange(q)
    Isignal = maskvol[dim/2-r:dim/2+r,dim/2-r:dim/2+r,dim/2-r:dim/2+r].ravel()
    signal = np.zeros(q,float)
    signal[Isignal] += amplitude
    X = np.zeros((n,p),float) + np.nan
    data = np.zeros((n,p),float) + np.nan
    vardata = np.zeros((n,p),float) + np.nan
    for i in xrange(n):
        X[i,I] = np.random.randn(len(I))
        if activation:
            o = np.array([dim/2,dim/2,dim/2])
            if jitter!=None:
                # numpy 2 casting rules don't allow in-place addition of float
                # and int.
                o = o + np.round(np.random.randn(3)*jitter).clip(r-mdim/2,mdim/2-r)
            #print o
        Ii = XYZvol[o[0]-r:o[0]+r,o[1]-r:o[1]+r,o[2]-r:o[2]+r].ravel()
        X[i,Ii] += amplitude
        vardata[i,I] = np.square(np.random.randn(len(I)))*noise**2
        data[i,I] = X[i,I] + np.random.randn(len(I))*np.sqrt(vardata[i,I])
    return data, XYZ, mask, XYZvol, vardata, signal

class test_displacement_field(unittest.TestCase):
    
    def test_sample_prior(self, verbose=False):
        data, XYZ, mask, XYZvol, vardata, signal = make_data(n=20, dim=20, r=3, mdim=15, maskdim=15, amplitude=5, noise=1, jitter=1, activation=True)
        D = displacement_field(XYZ, sigma=2.5, n=data.shape[0], mask=mask)
        B = len(D.block)
        for b in np.random.permutation(range(B)):
            for i in xrange(data.shape[0]):
                if verbose:
                    print 'sampling field', i, 'block', b
                U, V, L, W, I = D.sample(i, b, 'prior', 1)
                block = D.block[b]
                D.U[:, i, b] =  U
                D.V[:, i, block] = V
                D.W[:, i, L] = W
                D.I[i, L] = I
    
    def test_sample_rand_walk(self, verbose=False):
        data, XYZ, mask, XYZvol, vardata, signal = make_data(n=20, dim=20, r=3, mdim=15, maskdim=15, amplitude=5, noise=1, jitter=1, activation=True)
        D = displacement_field(XYZ, sigma=2.5*np.ones(3), n=data.shape[0], mask=mask)
        B = len(D.block)
        for b in np.random.permutation(range(B)):
            for i in xrange(data.shape[0]):
                if verbose:
                    print 'sampling field', i, 'block', b
                U, V, L, W, I = D.sample(i, b, 'rand_walk', 1e-2)
                block = D.block[b]
                D.U[:, i, b] =  U
                D.V[:, i, block] = V
                D.W[:, i, L] = W
                D.I[i, L] = I
    
    def test_sample_prior(self, verbose=False):
        data, XYZ, mask, XYZvol, vardata, signal = make_data(n=20, dim=20, r=3, mdim=15, maskdim=15, amplitude=5, noise=1, jitter=1, activation=True)
        D = displacement_field(XYZ, sigma=2.5, n=data.shape[0], mask=mask)
        B = len(D.block)
        for b in np.random.permutation(range(B)):
            for i in xrange(data.shape[0]):
                if verbose:
                    print 'sampling field', i, 'block', b
                U, V, L, W, I = D.sample(i, b, 'prior', 1)
                block = D.block[b]
                D.U[:, i, b] =  U
                D.V[:, i, block] = V
                D.W[:, i, L] = W
                D.I[i, L] = I
    
    def test_sample_all_blocks(self, verbose=False):
        data, XYZ, mask, XYZvol, vardata, signal = make_data(n=20, dim=20, r=3, mdim=15, maskdim=15, amplitude=5, noise=1, jitter=1, activation=True)
        D = displacement_field(XYZ, sigma=2.5, n=data.shape[0], mask=mask)
        for i in xrange(data.shape[0]):
            if verbose:
                print 'sampling field', i
            U, V, W, I = D.sample_all_blocks(1e-2)
            D.U[:, i] =  U
            D.V[:, i] = V
            D.W[:, i] = W
            D.I[i] = I

class test_gaussian_random_field(unittest.TestCase):
    
    def test_sample(self, verbose=False):
        data, XYZ, mask, XYZvol, vardata, signal = make_data(n=20, dim=20, r=3, mdim=15, maskdim=15, amplitude=5, noise=1, jitter=1, activation=True)
        n=data.shape[0]
        D = gaussian_random_field(XYZ, 2.5, n)
        for i in xrange(n):
            if verbose:
                print 'sampling field', i+1, 'out of', n
            U, V, L, W, I = D.sample(i, 1)
            D.U[:, i], D.V[:, i], D.W[:, i, L], D.I[i, L] = U, V, W, I

if __name__ == "__main__":
    unittest.main()
