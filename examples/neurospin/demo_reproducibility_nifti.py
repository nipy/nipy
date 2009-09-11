"""
Example of script to analyse the reproducibility in group studies
using a bootstrap procedure
author: Bertrand Thirion, 2005-2009
"""
import numpy as np
import os.path as op
import cPickle
import nifti
import tempfile
import get_data_light
from nipy.neurospin.utils.reproducibility_measures import \
     voxel_reproducibility, cluster_reproducibility, map_reproducibility

# -------------------------------------------------------
# -------- Set the paths, data, etc. --------------------
# -------------------------------------------------------

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
swd = tempfile.mkdtemp('nifti')

# -------------------------------------------------------
# ---------- Make a group mask --------------------------
# -------------------------------------------------------

# Read the masks
rmask = nifti.NiftiImage(mask_images[0])
ref_dim = rmask.getVolumeExtent()

mask = np.zeros(ref_dim)
for s in range(nsubj):
    rmask = nifti.NiftiImage(mask_images[s])
    m1 = rmask.asarray().T
    if (rmask.getVolumeExtent() != ref_dim):
        raise ValueError, "icompatible image size"
    mask += m1>0
        
# "intersect" the masks
mask = mask>nsubj/2
xyz = np.where(mask)
xyz = np.array(xyz).T
nvox = xyz.shape[0]

# -------------------------------------------------------
# ---------- load the functional images -----------------
# -------------------------------------------------------

# Load the betas
Functional = []
VarFunctional = []
tiny = 1.e-15
for s in range(nsubj): 
    beta = []
    varbeta = []
    rbeta = nifti.NiftiImage(contrast_images[s])
    temp = (rbeta.asarray().T)[mask]
    beta.append(temp)
    rbeta = nifti.NiftiImage(stat_images[s])
    tempstat = (rbeta.asarray().T)[mask]
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
Functional[np.isnan(Functional)]=0
VarFunctional[np.isnan(VarFunctional)]=0

# -------------------------------------------------------
# ---------- MNI coordinates ----------------------------
# -------------------------------------------------------

affine = rmask.header['sform']
coord = np.hstack((xyz, np.ones((nvox, 1))))
coord = np.dot(coord, affine.T)[:,:3]

# -------------------------------------------------------
# ---------- script ----------------------------
# -------------------------------------------------------

ngroups = 10
thresholds = [1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0]
sigma = 6.0
csize = 10
niter = 10
method = 'crfx'
verbose = 0

# BSA stuff
header = rmask.header
smin = 5
theta= 3.
dmax =  5.
tht = nsubj/4
thq = 0.9
afname = '/tmp/af'

swap = True

kap = []
clt = []
for threshold in thresholds:
    kappa = []
    cls = []
    if method=='rfx':
        kwargs = {'threshold':threshold}
    if method=='cffx':
        kwargs={'threshold':threshold,'csize':csize}
    if method=='crfx':
        kwargs={'threshold':threshold,'csize':csize}
    if method=='cmfx':
        kwargs={'threshold':threshold,'csize':csize}
    if method=='bsa':
        kwargs={'header':header,'smin':smin,'theta':theta,
                'dmax':dmax,'ths':ths,'thq':thq,'afname':afname}
        
    for i in range(niter):
        k = voxel_reproducibility(Functional, VarFunctional, xyz, ngroups,
                                  method, swap, verbose, **kwargs)
        kappa.append(k)
        cld = cluster_reproducibility(Functional, VarFunctional, xyz, ngroups,
                                       coord, sigma, method, swap, 
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


# -------------------------------------------------------
# ---------- create an image ----------------------------
# -------------------------------------------------------

th = 4.0
swap = True
kwargs = {'threshold':th,'csize':csize}
rmap = map_reproducibility(Functional, VarFunctional, xyz, ngroups,
                           method, swap, verbose, **kwargs)
wmap  = np.zeros(ref_dim).astype(np.int)
wmap[mask] = rmap
wim = nifti.NiftiImage(wmap.T,rbeta.header)
wim.description= 'reproducibility map at threshold %f, \
                 cluster size %d'%(th,csize)
wname = op.join(swd,'repro.nii')
wim.save(wname)

print('Wrote a reproducibility image in %s'%wname)


import two_binomial_mixture as mtb
MB = mtb.TwoBinomialMixture()
MB.estimate_parameters(rmap, ngroups+1)
h = np.array([np.sum(rmap==i) for i in range(ngroups+1)])
MB.show(h)
print MB.kappa()

mp.show()
