"""
Example of a script that uses the BSA (Bayesian Structural Analysis)
-- nipy.neurospin.spatial_models.bayesian_structural_analysis --
function

Please adapt the image paths to make it work on your own data

fixme: use for that some data on the www.

Author : Bertrand Thirion, 2008-2009
"""

#autoindent
import numpy as np
import scipy.stats as st
import os.path as op
import tempfile
import nifti

import nipy.neurospin.spatial_models.bayesian_structural_analysis as bsa
import nipy.neurospin.graph.field as ff
import get_data_light



def make_bsa_nifti(mask_images, betas, theta=3., dmax= 5., ths=0,
                    thq=0.5, smin=0, swd="/tmp/", nbru=None, nbeta=0):
    """
    main function for  performing bsa on a set of images
    """
    nbsubj = len(mask_images)
    if nbru==None:
        bru = range(nbsubj)
    
    # Read the referential
    nim = nifti.NiftiImage(mask_images[0])
    header = nim.header
    ref_dim = nim.getVolumeExtent()
    grid_size = np.prod(ref_dim)
    sform = nim.header['sform']
    voxsize = nim.getVoxDims()

    # Read the masks and compute the "intersection"
    mask = np.zeros(ref_dim)
    for s in range(nbsubj):
        nim = nifti.NiftiImage(mask_images[s])
        temp = nim.asarray().T
        mask = mask+temp;

    xyz = np.array(np.where(mask>nbsubj/2))
    nbvox = np.size(xyz,1)

    # create the field strcture that encodes image topology
    Fbeta = ff.Field(nbvox)
    Fbeta.from_3d_grid(xyz.astype(np.int).T,18)

    # Get  coordinates in mm
    xyz = np.transpose(xyz)
    xyz = np.hstack((xyz,np.ones((nbvox,1))))
    tal = np.dot(xyz,np.transpose(sform))[:,:3]
    xyz = xyz.astype(np.int)
    
    # read the functional images
    lbeta = []
    for s in range(nbsubj):
        rbeta = nifti.NiftiImage(betas[s])
        beta = rbeta.asarray().T
        beta = beta[mask>nbsubj/2]
        lbeta.append(beta)
        
    lbeta = np.array(lbeta).T

    g0 = 1.0/(np.prod(voxsize)*nbvox)
    bdensity = 1

    # choose the method  you prefer
    crmap = np.zeros(nbvox)
    p = np.zeros(nbvox)
    AF = None
    BF = [None for s in range(nbsubj)]
    #crmap,AF,BF,p = bsa.compute_BSA_ipmi(Fbeta,lbeta,tal,dmax,xyz[:,:3],
    #                                     header,thq, smin,ths, theta,g0,
    #                                     bdensity)
    #crmap,AF,BF,p = bsa.compute_BSA_dev (Fbeta,lbeta,tal,dmax,xyz[:,:3],
    #                                     header,thq, smin,ths, theta,g0,
    #                                     bdensity,verbose=1)
    crmap,AF,BF,p = bsa.compute_BSA_simple (Fbeta,lbeta,tal,dmax,xyz[:,:3],
                                            header,thq, smin,ths, theta,g0,
                                            verbose=0)

    # Write the results
    LabelImage = op.join(swd,"CR_%04d.nii"%nbeta)
    Label = -2*np.ones(ref_dim,'int16')
    Label[mask>nbsubj/2] = crmap.astype('i')
    nim = nifti.NiftiImage(np.transpose(Label),rbeta.header)    
    nim.description='group Level labels from bsa procedure'
    nim.save(LabelImage)    
    
    if bdensity:
        DensImage = op.join(swd,"density_%04d.nii"%nbeta)
        density = np.zeros(ref_dim)
        density[mask>nbsubj/2]=p
        nim = nifti.NiftiImage(np.transpose(density),rbeta.header)
        nim.description='group-level spatial density of active regions'
        nim.save(DensImage)
        
    for s in range(nbsubj):
        LabelImage = op.join(swd,"AR_s%04d_%04d.nii"%(nbru[s],nbeta))
        Label = -2*np.ones(ref_dim,'int16')
        Label[mask>nbsubj/2]=-1
        if BF[s]!=None:
            nls = BF[s].get_roi_feature('label')
            nls[nls==-1] = AF.k+2
            for k in range(BF[s].k):
                xyzk = BF[s].xyz[k].T 
                Label[xyzk[0],xyzk[1],xyzk[2]] =  nls[k]
        
        nim = nifti.NiftiImage(np.transpose(Label),rbeta.header)
        nim.description='Individual label image from bsa procedure'
        nim.save(LabelImage)
        

    return AF,BF


# Get the data
get_data_light.getIt()
nbsubj = 12
nbeta = 29
data_dir = op.expanduser(op.join('~', '.nipy', 'tests', 'data',
                                 'group_t_images'))
mask_images = [op.join(data_dir,'mask_subj%02d.nii'%n)
               for n in range(nbsubj)]

betas =[ op.join(data_dir,'spmT_%04d_subj_%02d.nii'%(nbeta,n))
                 for n in range(nbsubj)]

# set various parameters
nbru = range(12)
theta = float(st.t.isf(0.01,100))
dmax = 5.
ths = 2 # or nbsubj/4
thq = 0.9
verbose = 1
smin = 5
swd = tempfile.mkdtemp()

AF,BF = make_bsa_nifti(mask_images, betas, theta, dmax,
                       ths,thq,smin,swd,nbru, nbeta)

# Write the result. OK, this is only a temporary solution
import pickle
picname = op.join(swd,"AF_%04d.pic" %nbeta)
pickle.dump(AF, open(picname, 'w'), 2)
picname = op.join(swd,"BF_%04d.pic" %nbeta)
pickle.dump(BF, open(picname, 'w'), 2)

print "Wrote all the stuff in %s"%swd
