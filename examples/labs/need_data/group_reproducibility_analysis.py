# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Example of script to analyse the reproducibility in group studies using a
bootstrap procedure.

This reproduces approximately the work described in 'Analysis of a large fMRI
cohort: Statistical and methodological issues for group analyses' Thirion B,
Pinel P, Meriaux S, Roche A, Dehaene S, Poline JB.  Neuroimage. 2007
Mar;35(1):105-20.

Needs matplotlib

Author: Bertrand Thirion, 2005-2009
"""

import os
import os.path as op

from numpy import array

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

from nipy.labs.utils.reproducibility_measures import \
     group_reproducibility_metrics

# Local import
from get_data_light import DATA_DIR, get_second_level_dataset

print 'This analysis takes a long while, please be patient'

##############################################################################
# Set the paths, data, etc.
##############################################################################

nsubj = 12
subj_id = range(nsubj)
nbeta = 29

data_dir = op.join(DATA_DIR, 'group_t_images')

mask_images = [op.join(data_dir, 'mask_subj%02d.nii' % n)
               for n in range(nsubj)]
stat_images = [op.join(data_dir, 'spmT_%04d_subj_%02d.nii' % (nbeta, n))
                 for n in range(nsubj)]
contrast_images = [op.join(data_dir, 'con_%04d_subj_%02d.nii' % (nbeta, n))
                 for n in range(nsubj)]
all_images = mask_images + stat_images + contrast_images
missing_file = array([not op.exists(m) for m in all_images]).any()

if missing_file:
    get_second_level_dataset()

swd = os.getcwd()

##############################################################################
# main script
##############################################################################

ngroups = [4]
thresholds = [3.0, 4.0, 5.0]
sigma = 6.0
csize = 10
niter = 10
method = 'crfx'
verbose = 0
swap = False

voxel_results, cluster_results, peak_results = group_reproducibility_metrics(
    mask_images, contrast_images, [], thresholds, ngroups, method,
    cluster_threshold=csize, number_of_samples=niter, sigma=sigma,
    do_clusters=True, do_voxels=True, do_peaks=True, swap=swap)

kap = [k for k in voxel_results[ngroups[0]].values()]
clt = [k for k in cluster_results[ngroups[0]].values()]
pk = [k for k in peak_results[ngroups[0]].values()]

##############################################################################
# plot
##############################################################################

plt.figure()
plt.subplot(1, 3, 1)
plt.boxplot(kap)
plt.title('voxel-level reproducibility')
plt.xticks(range(1, 1 + len(thresholds)), thresholds)
plt.xlabel('threshold')
plt.subplot(1, 3, 2)
plt.boxplot(clt)
plt.title('cluster-level reproducibility')
plt.xticks(range(1, 1 + len(thresholds)), thresholds)
plt.xlabel('threshold')
plt.subplot(1, 3, 3)
plt.boxplot(clt)
plt.title('cluster-level reproducibility')
plt.xticks(range(1, 1 + len(thresholds)), thresholds)
plt.xlabel('threshold')


##############################################################################
# create an image
##############################################################################
"""
# this is commented until a new version of the code allows it
# with the adequate level of abstraction
th = 4.0
swap = False
kwargs = {'threshold':th,'csize':csize}
rmap = map_reproducibility(Functional, VarFunctional, grp_mask, ngroups,
                           method, swap, verbose, **kwargs)
wmap  = mask.astype(np.int)
wmap[mask] = rmap
wim = Nifti1Image(wmap, affine)
wim.get_header()['descrip']= 'reproducibility map at threshold %f, \
                             cluster size %d'%(th,csize)
wname = op.join(swd,'repro.nii')
save(wim, wname)

print('Wrote a reproducibility image in %s'%wname)

"""
