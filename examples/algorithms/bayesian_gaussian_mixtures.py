# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Example of a demo that fits a Bayesian Gaussian Mixture Model (GMM) 
to  a dataset.

Variational bayes and Gibbs estimation are sucessively run on the same
dataset


Author : Bertrand Thirion, 2008-2010
"""
print __doc__

import numpy as np
import numpy.random as nr
import pylab as pl

import nipy.algorithms.clustering.bgmm as bgmm
from nipy.algorithms.clustering.gmm import plot2D


dim = 2

###############################################################################
# 1. generate a 3-components mixture
x1 = nr.randn(25, dim)
x2 = 3 + 2 * nr.randn(15, dim)
x3 = np.repeat(np.array([-2, 2], ndmin=2), 10, 0) + 0.5 * nr.randn(10, dim)
x = np.concatenate((x1, x2, x3))

###############################################################################
#2. fit the mixture with a bunch of possible models, using Variational Bayes
krange = range(1, 10)
be = - np.inf
for  k in krange:
    b = bgmm.VBGMM(k, dim)
    b.guess_priors(x)
    b.initialize(x)
    b.estimate(x)
    ek = float(b.evidence(x))
    if ek > be:
        be = ek
        bestb = b
        
    print k, 'classes, free energy:', ek

###############################################################################
# 3. plot the result
z = bestb.map_label(x)
plot2D(x, bestb, z, verbose=0)
pl.title('Variational Bayes')

###############################################################################
# 4. the same, with the Gibbs GMM algo
niter = 1000
krange = range(1, 6)
bbf = - np.inf
for k in krange:
    b = bgmm.BGMM(k, dim)
    b.guess_priors(x)
    b.initialize(x)
    b.sample(x, 100)
    w, cent, prec, pz = b.sample(x, niter=niter, mem=1)
    bplugin = bgmm.BGMM(k, dim, cent, prec, w)
    bplugin.guess_priors(x)
    bfk = bplugin.bayes_factor(x, pz.astype(np.int), nperm=120)
    print k, 'classes, evidence:', bfk
    if bfk > bbf:
        bestk = k
        bbf = bfk
        bbgmm = bplugin

z = bbgmm.map_label(x)
plot2D(x, bbgmm, z, verbose=0)
pl.title('Gibbs sampling')
pl.show()
