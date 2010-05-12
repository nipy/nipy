# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Example of script to analyse the reproducibility in group studies
using a bootstrap procedure

This reproduces approximately the work described in
Analysis of a large fMRI cohort: Statistical and methodological issues for group analyses.
Thirion B, Pinel P, Meriaux S, Roche A, Dehaene S, Poline JB.
Neuroimage. 2007 Mar;35(1):105-20. 

author: Bertrand Thirion, 2005-2009
"""
import numpy as np
import os.path as op
import cPickle
from nipy.io.imageformats import load, save, Nifti1Image 
import tempfile
import get_data_light

from nipy.neurospin.utils.mask import intersect_masks
from nipy.neurospin.utils.reproducibility_measures import \
     voxel_reproducibility, cluster_reproducibility, map_reproducibility

print 'This analysis takes a long while, please be patient'

################################################################################
# Set the paths, data, etc.

get_data_light.getIt()
nsubj = 12
subj_id = range(nsubj)
nbeta = 29
data_dir = op.expanduser(op.join('~', '.nipy', 'tests', 'data',
                                 'group_t_images'))
mask_images = [op.join(data_dir,'mask_subj%02d.nii'%n)
               for n in range(nsubj)]

stat_images =[ op.join(data_dir,'spmT_%04d_subj_%02d.nii'%(nbeta,n))
                 for n in range(nsubj)]
contrast_images =[ op.join(data_dir,'con_%04d_subj_%02d.nii'%(nbeta,n))
                 for n in range(nsubj)]
swd = tempfile.mkdtemp('image')

################################################################################
# Make a group mask

affine = load(mask_images[0]).get_affine()
mask = intersect_masks(mask_images)>0
grp_mask = Nifti1Image(mask, affine)
xyz = np.where(mask)
xyz = np.array(xyz).T
nvox = xyz.shape[0]

################################################################################
# Load the functional images

# Load the betas
Functional = []
VarFunctional = []
tiny = 1.e-15
for s in range(nsubj): 
    beta = []
    varbeta = []
    rbeta = load(contrast_images[s])
    temp = (rbeta.get_data())[mask]
    beta.append(temp)
    rbeta = load(stat_images[s])
    tempstat = (rbeta.get_data())[mask]
    # ugly trick to get the variance
    varbeta = temp**2/(tempstat**2+tiny)

    varbeta = np.array(varbeta)
    beta = np.array(beta)
    Functional.append(beta.T)
    VarFunctional.append(varbeta.T)
    
Functional = np.array(Functional)
Functional = np.squeeze(Functional).T
VarFunctional = np.array(VarFunctional)
VarFunctional = np.squeeze(VarFunctional).T
Functional[np.isnan(Functional)] = 0
VarFunctional[np.isnan(VarFunctional)] = 0


################################################################################
# script

ngroups = 10
thresholds = [1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0]
sigma = 6.0
csize = 10
niter = 10
method = 'crfx'
verbose = 0

# apply random sign swaps to the data to test the reproducibility
# under the H0 hypothesis
swap = False

kap = []
clt = []
for threshold in thresholds:
    kappa = []
    cls = []
    kwargs={'threshold':threshold,'csize':csize}
        
    for i in range(niter):
        k = voxel_reproducibility(Functional, VarFunctional, grp_mask, ngroups,
                                  method, swap, verbose, **kwargs)
        kappa.append(k)
        cld = cluster_reproducibility(Functional, VarFunctional, grp_mask, ngroups,
                                       sigma, method, swap, 
                                      verbose, **kwargs)
        cls.append(cld)
        
    kap.append(np.array(kappa))
    clt.append(np.array(cls))
    
import matplotlib.pylab as mp
mp.figure()
mp.subplot(1,2,1)
mp.boxplot(kap)
mp.title('voxel-level reproducibility')
mp.xticks(range(1,1+len(thresholds)),thresholds)
mp.xlabel('threshold')
mp.subplot(1,2,2)
mp.boxplot(clt)
mp.title('cluster-level reproducibility')
mp.xticks(range(1,1+len(thresholds)),thresholds)
mp.xlabel('threshold')


################################################################################
# create an image

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

