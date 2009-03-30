"""
Example of a script that uses the BSA function
Please adapt the image paths to make it work on your own data
"""

#autoindent
import numpy as np
import scipy.stats as st
import os.path as op

import fff2.spatial_models.bayesian_structural_analysis as bsa
import fff2.graph.field as ff
import nifti


def make_bsa_nifti(nbsubj, Mask_Images, betas, nbru='1', theta=3., dmax =  5., ths = 0, thq = 0.5, smin = 0, swd = "/tmp/",nbeta = [0]):
	"""
	main function for  performing bsa on a set of images
	"""
	
	# Read the referential
	nim = nifti.NiftiImage(Mask_Images[0])
	ref_dim = nim.getVolumeExtent()
	grid_size = np.prod(ref_dim)
	sform = nim.header['sform']
	voxsize = nim.getVoxDims()

	# Read the masks and compute the "intersection"
	mask = np.zeros(ref_dim)
	for s in range(nbsubj):
		nim = nifti.NiftiImage(Mask_Images[s])
		temp = np.transpose(nim.getDataArray())
		mask = mask+temp;

	xyz = np.array(np.where(mask>nbsubj/2))
	nbvox = np.size(xyz,1)

	# create the field strcture that encodes image topology
	Fbeta = ff.Field(nbvox)
	Fbeta.from_3d_grid(np.transpose(xyz.astype('i')),18)

	# Get  coordinates in mm
	xyz = np.transpose(xyz)
	xyz = np.hstack((xyz,np.ones((nbvox,1))))
	tal = np.dot(xyz,np.transpose(sform))[:,:3]
	xyz = xyz.astype('i')
	
	# read the functional images
	lbeta = []
	for s in range(nbsubj):
		rbeta = nifti.NiftiImage(betas[s][0])
		beta = np.transpose(rbeta.getDataArray())
		beta = beta[mask>nbsubj/2]
		lbeta.append(beta)
		
	lbeta = np.transpose(np.array(lbeta))

	g0 = 1.0/(np.prod(voxsize)*nbvox)
	bdensity = 1

	# choose the method  you prefer
	
	#crmap,AF,BF,u,p = bsa.compute_BSA_ipmi(Fbeta,lbeta,tal,dmax,thq, smin,ths, theta,g0,bdensity)
	#crmap,AF,BF,u,p = bsa.compute_BSA_dev (Fbeta,lbeta,tal,dmax,thq, smin,ths, theta,g0,bdensity)
	crmap,AF,BF,u,p = bsa.compute_BSA_simple (Fbeta,lbeta,tal,dmax,thq, smin,ths, theta,g0,bdensity)

	# Write the results
	LabelImage = op.join(swd,"CR_%04d.nii"%nbeta[0])
	Label = -2*np.ones(ref_dim,'int16')
	Label[mask>nbsubj/2] = crmap.astype('i')
	nim = nifti.NiftiImage(np.transpose(Label),rbeta.header)	
	nim.description='group Level labels from bsa procedure'
	nim.save(LabelImage)	
	
	u[u==-1] = np.size(AF)+2

	if bdensity:
		DensImage = op.join(swd,"density_%04d.nii"%nbeta[0])
		density = np.zeros(ref_dim)
		density[mask>nbsubj/2]=p
		nim = nifti.NiftiImage(np.transpose(density),rbeta.header)
		nim.description='group-level spatial density of active regions'
		nim.save(DensImage)
		
	qq = 0
	sub = np.concatenate([s*np.ones(BF[s].k) for s in range(nbsubj)])
	for s in range(nbsubj):
		LabelImage = op.join(swd,"AR_s%04d_%04d.nii"%(nbru[s],nbeta[0]))
		Label = -2*np.ones(ref_dim,'int16')
		lw = BF[s].label.astype('i')
		us = u[sub==s]
		lw[lw>-1]= us[lw[lw>-1]]
		#print s, lw.max()
		Label[mask>nbsubj/2] = lw
		nim = nifti.NiftiImage(np.transpose(Label),rbeta.header)
		nim.description='Individual label image from bsa procedure'
		nim.save(LabelImage)
		qq = qq + BF[s].get_k()

	return AF,BF


# Get the data
nbru = range(1,13)

Sess = len(nbru)
nbeta = [29]
theta = float(st.t.isf(0.01,100))
dmax = 5.
ths = Sess/2-1
thq = 0.9
verbose = 1
smin = 5

swd = "/tmp/"

# a mask of the brain in each subject
Mask_Images =["/volatile/thirion/Localizer/sujet%02d/functional/fMRI/spm_analysis_RNorm_S/mask.img" % bru for bru in nbru]

# activation image in each subject
betas = [["/volatile/thirion/Localizer/sujet%02d/functional/fMRI/spm_analysis_RNorm_S/spmT_%04d.img" % (bru, n) for n in nbeta] for bru in nbru]

AF,BF = make_bsa_nifti(Sess, Mask_Images, betas, nbru, theta, dmax, ths,thq,smin,swd,nbeta)

# xrite the result. OK, this is only a temporary solution
import cPickle
picname = op.join(swd,"AF_%04d" %nbeta[0])
cPickle.dump(AF, open(picname, 'w'), 2)
picname = op.join(swd,"BF_%04d" %nbeta[0])
cPickle.dump(BF, open(picname, 'w'), 2)

