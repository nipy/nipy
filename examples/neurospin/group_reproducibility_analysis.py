"""
Example of script to analyse the reproducibility in group studies
using a bootstrap procedure

author: Bertrand Thirion, 2005-2009
"""
print __doc__

import numpy as np

import nipy.neurospin.utils.simul_2d_multisubject_fmri_dataset as simul
from nipy.neurospin.utils.reproducibility_measures import \
     voxel_reproducibility, cluster_reproducibility, map_reproducibility,\
     peak_reproducibility


################################################################################
# Generate the data 
nsubj = 105
dimx = 60
dimy = 60
pos = 2*np.array([[ 6,  7],
                  [10, 10],
                  [15, 10]])
ampli = 0.5* np.array([5, 7, 6])
dataset = simul.make_surrogate_array(nbsubj=nsubj, dimx=dimx, dimy=dimy, 
                                     pos=pos, ampli=ampli, width=5.0)
betas = np.reshape(dataset, (nsubj, dimx, dimy))

# set the variance at 1 everywhere
func = np.reshape(betas,(nsubj, dimx*dimy)).T
var = np.ones((dimx*dimy, nsubj))
xyz = np.array(np.where(betas[:1])).T
coord = xyz.astype(np.float)

################################################################################
# Run reproducibility analysis 

ngroups = 10
thresholds = [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
sigma = 2.0
csize = 10
niter = 10
method = 'crfx'
verbose = 0
swap = False#True

 
kap = []
clt = []
pk = []
sens = []
for threshold in thresholds:
    kwargs={'threshold':threshold,'csize':csize}
    kappa = []
    cls = []
    sent = []
    peaks = []
    for i in range(niter):
        k = voxel_reproducibility(func, var, xyz, ngroups,
                                  method, swap, verbose, **kwargs)
        kappa.append(k)
        cld = cluster_reproducibility(func, var, xyz, ngroups, coord, sigma,
                                      method, swap, verbose, **kwargs)
        cls.append(cld)
        peak = peak_reproducibility(func, var, xyz, ngroups, coord, sigma,
                                      method, swap, verbose, **kwargs)
        peaks.append(peak)
        seni = map_reproducibility(func, var, xyz, ngroups,
                           method, True, verbose, threshold=threshold,
                           csize=csize).mean()/ngroups
        sent.append(seni)
    sens.append(np.array(sent))
    kap.append(np.array(kappa))
    clt.append(np.array(cls))
    pk.append(np.array(peaks))
    
################################################################################
# Visualize the results
import scipy.stats as st
aux = st.norm.sf(np.array(thresholds))#,nsubj/ngroups)

import matplotlib.pylab as mp
a = mp.figure()
mp.subplot(1,2,1)
mp.boxplot(kap)
mp.boxplot(sens)
# mp.plot(aux)
mp.title('voxel-level reproducibility', fontsize=12)
mp.xticks(range(1,1+len(thresholds)),thresholds)
mp.xlabel('threshold')
mp.subplot(1,2,2)
mp.boxplot(clt)
mp.boxplot(pk,notch=1)
mp.title('cluster-level reproducibility', fontsize=12)
mp.xticks(range(1,1+len(thresholds)),thresholds)
mp.xlabel('threshold')
a.set_figwidth(10.)


mp.figure()
q = 1
for threshold in thresholds:
    mp.subplot(3, len(thresholds)/3, q)
    rmap = map_reproducibility(func, var, xyz, ngroups,
                           method, verbose, threshold=threshold,
                           csize=csize)
    rmap = np.reshape(rmap, (dimx, dimy))
    mp.imshow(rmap, interpolation=None, vmin=0, vmax=ngroups)
    mp.title('threshold: %f' % threshold, fontsize=10)
    mp.axis('off')
    q +=1

mp.suptitle('Map reproducibility for different thresholds') 
#mp.colorbar()
mp.show()

