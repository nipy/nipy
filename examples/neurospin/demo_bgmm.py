"""
Example of a demo that fits a Bayesian GMM to  a dataset


Author : Bertrand Thirion, 2008-2009
"""

import numpy as np
import numpy.random as nr
import nipy.neurospin.clustering.bgmm as bgmm
from nipy.neurospin.clustering.gmm import plot2D


dim = 2
# 1. generate a 3-components mixture
x1 = nr.randn(100,dim)
x2 = 3+2*nr.randn(50,dim)
x3 = np.repeat(np.array([-2,2],ndmin=2),30,0)+0.5*nr.randn(30,dim)
x = np.concatenate((x1,x2,x3))

#2. fit the mixture with a bunch of possible models
krange = range(1,10)
be = -np.infty
for  k in krange:
    b = bgmm.VBGMM(k,dim)
    b.guess_priors(x)
    b.initialize(x)
    b.estimate(x)
    ek = b.evidence(x)
    if ek>be:
        be = ek
        bestb = b
        
    print k,'classes, free energy:',b.evidence(x)

# 3, plot the result
z = bestb.map_label(x)
plot2D(x,bestb,z,show=1,verbose=0)

# the same, with the Gibbs GMM algo
niter = 1000
krange = range(2,5)
bbf = -np.infty
for k in range(1,4):
    b = bgmm.BGMM(k,dim)
    b.guess_priors(x)
    b.initialize(x)
    b.sample(x,100)
    w,cent,prec,pz = b.sample(x,niter=niter,mem=1)
    bplugin =  bgmm.BGMM(k,dim,cent,prec,w)
    bplugin.guess_priors(x)
    bfk = bplugin.Bfactor(x,pz.astype(np.int),1)
    print k, 'classes, evidence:',bfk
    if bfk>bbf:
        bestk = k
        bbf = bfk

z = bplugin.map_label(x)
plot2D(x,bplugin,z,show=1,verbose=0)

