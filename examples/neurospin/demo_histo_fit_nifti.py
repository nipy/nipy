"""
Example of a script that perfoms histogram analysis of an activation image.
This is based on a real fMRI image

Simply modify the input image path to make it work on your preferred
nifti image

Author : Bertrand Thirion, 2008-2009
"""

import numpy as np
import os
import nifti
import scipy.stats as st

import nipy.neurospin.utils.emp_null as en

swd = "/tmp/"
verbose = 1

data_dir = os.path.expanduser(os.path.join('~', '.nipy', 'tests', 'data'))
MaskImage = os.path.join(data_dir,'mask.nii.gz')
InputImage = os.path.join(data_dir,'spmT_0029.nii.gz')

if os.path.exists(InputImage)==False:
    import urllib2
    url = 'ftp://ftp.cea.fr/pub/dsv/madic/download/nipy'
    filename = 'mask.nii.gz'
    datafile = os.path.join(url,filename)
    fp = urllib2.urlopen(datafile)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        assert os.path.exists(data_dir)
    local_file = open(MaskImage, 'w')
    local_file.write(fp.read())
    local_file.flush()
    local_file.close()
    filename = 'spmT_0029.nii.gz'
    datafile = os.path.join(url,filename)
    fp = urllib2.urlopen(datafile)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        assert os.path.exists(data_dir)
    local_file = open(InputImage, 'w')
    local_file.write(fp.read())
    local_file.flush()
    local_file.close()   

        



theta = float(st.t.isf(0.01,100))

# Read the referential
nim = nifti.NiftiImage(MaskImage)
ref_dim = nim.getVolumeExtent()
grid_size = np.prod(ref_dim)
sform = nim.header['sform']
voxsize = nim.getVoxDims()

# Read the masks and compute the "intersection"
mask = nim.asarray().T
xyz = np.array(np.where(mask))
nbvox = np.size(xyz,1)

# read the functional image
rbeta = nifti.NiftiImage(InputImage)
beta = rbeta.asarray().T
beta = beta[mask>0]

# fit beta's histogram with a Gamma-Gaussian mixture
bfm = np.array([2.5,3.0,3.5,4.0,4.5])
bfp = en.Gamma_Gaussian_fit(np.squeeze(beta),bfm,verbose=2)

# fit beta's histogram with a mixture of Gaussians
alpha = 0.01
prior_strength = 100
bfq = en.three_classes_GMM_fit(beta, bfm, alpha, prior_strength,verbose=2)

# fit the null mode of beta with the robust method
efdr = en.ENN(beta)
efdr.learn()
efdr.plot(bar=0)
