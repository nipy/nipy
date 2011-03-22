# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Example of script to analyse the reproducibility in group studies
using a bootstrap procedure

author: Bertrand Thirion, 2005-2009
"""
print __doc__

import numpy as np

import nipy.labs.utils.simul_multisubject_fmri_dataset as simul
from nipy.labs.utils.reproducibility_measures import \
     voxel_reproducibility, cluster_reproducibility, map_reproducibility,\
     peak_reproducibility


###############################################################################
# Generate the data 
nsubj = 105
dimx = 60
dimy = 60
pos = np.array([[12, 14],
                [20, 20],
                [30, 20]])
ampli = np.array([2.5, 3.5, 3])
dataset = simul.surrogate_2d_dataset(nbsubj=nsubj, dimx=dimx, dimy=dimy, 
                                     pos=pos, ampli=ampli, width=5.0)
betas = np.reshape(dataset, (nsubj, dimx, dimy))

# set the variance at 1 everywhere
func = np.reshape(betas, (nsubj, dimx * dimy)).T
var = np.ones((dimx * dimy, nsubj))

from nibabel import Nifti1Image
mask = Nifti1Image(np.ones((dimx, dimy, 1)), np.eye(4))

###############################################################################
# Run reproducibility analysis 

ngroups = 10
thresholds = np.arange(.5, 6., .5)
sigma = 2.0
csize = 10
niter = 10
method = 'crfx'
verbose = 0

# do not use permutations
swap = False

 
kap = []
clt = []
pk = []
sens = []
for threshold in thresholds:
    kwargs={'threshold': threshold, 'csize': csize}
    kappa = []
    cls = []
    sent = []
    peaks = []
    for i in range(niter):
        k = voxel_reproducibility(func, var, mask, ngroups,
                                  method, swap, verbose, **kwargs)
        kappa.append(k)
        cld = cluster_reproducibility(func, var, mask, ngroups, sigma,
                                      method, swap, verbose, **kwargs)
        cls.append(cld)
        peak = peak_reproducibility(func, var, mask, ngroups, sigma,
                                      method, swap, verbose, **kwargs)
        peaks.append(peak)
        seni = map_reproducibility(func, var, mask, ngroups,
                           method, True, verbose, threshold=threshold,
                           csize=csize).mean()/ngroups
        sent.append(seni)
    sens.append(np.array(sent))
    kap.append(np.array(kappa))
    clt.append(np.array(cls))
    pk.append(np.array(peaks))
    
###############################################################################
# Visualize the results
import scipy.stats as st
aux = st.norm.sf(thresholds)

import matplotlib.pylab as mp
a = mp.figure()
mp.subplot(1, 3, 1)
mp.boxplot(kap)
mp.title('voxel-level reproducibility', fontsize=12)
mp.xticks(range(1, 1 + len(thresholds)), thresholds)
mp.xlabel('threshold')
mp.subplot(1, 3, 2)
mp.boxplot(clt)
mp.title('cluster-level reproducibility', fontsize=12)
mp.xticks(range(1, 1 + len(thresholds)), thresholds)
mp.xlabel('threshold')
mp.subplot(1, 3, 3)
mp.boxplot(pk, notch=1)
mp.title('peak-level reproducibility', fontsize=12)
mp.xticks(range(1, 1 + len(thresholds)), thresholds)
mp.xlabel('threshold')
a.set_figwidth(10.)
a.set_size_inches(12, 5)

mp.figure()
for q, threshold in enumerate(thresholds):
    mp.subplot(3, len(thresholds) / 3 + 1, q + 1)
    rmap = map_reproducibility(func, var, mask, ngroups,
                           method, verbose, threshold=threshold,
                           csize=csize)
    rmap = np.reshape(rmap, (dimx, dimy))
    mp.imshow(rmap, interpolation=None, vmin=0, vmax=ngroups)
    mp.title('threshold: %f' % threshold, fontsize=10)
    mp.axis('off')


mp.suptitle('Map reproducibility for different thresholds') 
mp.show()
