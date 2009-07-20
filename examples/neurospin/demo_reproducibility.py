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

# Get the data
get_data_light.getIt()
nbsubj = 12
subj_id = range(nbsubj)
nbeta = 29
data_dir = op.expanduser(op.join('~', '.nipy', 'tests', 'data',
                                 'group_t_images'))
mask_images = [op.join(data_dir,'mask_subj%02d.nii'%n)
               for n in range(nbsubj)]

stat_images =[ op.join(data_dir,'spmT_%04d_subj_%02d.nii'%(nbeta,n))
                 for n in range(nbsubj)]
contrast_images =[ op.join(data_dir,'con_%04d_subj_%02d.nii'%(nbeta,n))
                 for n in range(nbsubj)]


# -------------------------------------------------------
# ---------- Make a group mask --------------------------
# -------------------------------------------------------

# Read the masks
rmask = nifti.NiftiImage(mask_images[0])
ref_dim = rmask.getVolumeExtent()

mask = np.zeros(ref_dim)
for s in range(nbsubj):
    rmask = nifti.NiftiImage(mask_images[s])
    m1 = rmask.asarray().T
    if (rmask.getVolumeExtent() != ref_dim):
        raise ValueError, "icompatible image size"
    mask += m1>0
        
# "intersect" the masks
mask = mask>nbsubj/2
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
for s in range(nbsubj): 
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

sform = rmask.header['sform']
coord = np.hstack((xyz, np.ones((nvox, 1))))
coord = np.dot(coord, sform.T)[:,:3]

groupsize = 12
thresholds = [2.0,2.5,3.0,3.5,4.0]
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
tht = groupsize/4
thq = 0.9
afname = '/tmp/af'


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
        k = voxel_reproducibility(Functional, VarFunctional, xyz, groupsize,
                                  method,verbose,**kwargs)
        kappa.append(k)
        cld = cluster_reproducibility(Functional, VarFunctional, xyz,
                                      groupsize, coord, sigma,method,
                                      verbose, **kwargs)
        cls.append(cld)
        print threshold,cld
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
mp.show()

import pickle
picname = '/tmp/'+'cluster_repro_%s.pic'%method
pickle.dump(clt, open(picname, 'w'), 2)
#toto = pickle.load(open(picname, 'r'))

rmap = map_reproducibility(Functional, VarFunctional, xyz, groupsize,
                                  method, verbose, **kwargs)

wmap  = np.zeros(ref_dim).astype(np.int)
wmap[mask]=rmap
wim = nifti.NiftiImage(wmap.T,rbeta.header)
wim.description('reproducibility map')
wim.save('/tmp/repro.nii')
