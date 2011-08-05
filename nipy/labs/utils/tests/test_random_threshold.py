# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import unittest

import numpy as np
import scipy.stats as st

from ..random_threshold import randthresh_main

def make_data(n=10, dim=20, r=5, mdim=15, maskdim=20, amplitude=10, 
                            noise=1, jitter=None, activation=False):
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
                # numpy 2 casting rules do no allow in place addition of ints
                # and floats - hence not in-place here
                o = o + np.round(np.random.randn(3)*jitter).clip(r-mdim/2,mdim/2-r)
            #print o
        Ii = XYZvol[o[0]-r:o[0]+r,o[1]-r:o[1]+r,o[2]-r:o[2]+r].ravel()
        X[i,Ii] += amplitude
        vardata[i,I] = np.square(np.random.randn(len(I)))*noise**2
        data[i,I] = X[i,I] + np.random.randn(len(I))*np.sqrt(vardata[i,I])
    return data, XYZ, mask, XYZvol, vardata, signal

class TestRandomThreshold(unittest.TestCase):
    
    def test_random_threshold(self):
         # Just run all random threshold functions on toy data
         # for smoke testing
         data, XYZ, mask, XYZvol, vardata, signal = make_data(n=1, dim=20, r=3, mdim=20, maskdim=20, amplitude=4, noise=0, jitter=0, activation=True)
         Y = data[0]
         X = np.clip(-np.log(1 - st.chi2.cdf(Y**2, 1, 0)), 0, 1e10)
         K = (signal == 0).sum() - 100
         verbose=False
         randthresh_main(X, K, XYZ=None, p=np.inf, varwind=True, 
                         knownull=True, stop=False, verbose=verbose)
         randthresh_main(Y, K, XYZ=None, p=np.inf, varwind=True,
                         knownull=False, stop=True, verbose=verbose)
         
         randthresh_main(X, K, XYZ=None, p=np.inf, varwind=False, 
                         knownull=True, stop=True, verbose=verbose)
         randthresh_main(Y, K, XYZ=None, p=np.inf, varwind=False,
                         knownull=False, stop=False, verbose=verbose)
         
         randthresh_main(X, K, XYZ=XYZ, p=np.inf, varwind=True,
                         knownull=True, stop=False, verbose=verbose)
         randthresh_main(Y, K, XYZ=XYZ, p=np.inf, varwind=True,
                         knownull=False, stop=False, verbose=verbose)
         
         randthresh_main(X, K, XYZ=XYZ, p=np.inf, varwind=False,
                         knownull=True, stop=True, verbose=verbose)
         randthresh_main(Y, K, XYZ=XYZ, p=np.inf, varwind=False,
                         knownull=False, stop=True, verbose=verbose)


if __name__ == "__main__":
    unittest.main()

