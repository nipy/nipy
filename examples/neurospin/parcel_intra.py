"""
Example of script to parcellate the data from one subject
And of various processing that can be applied then
author: Bertrand Thirion, 2005-2009
"""


import numpy as np
import fff2.graph as fg
from fff2.clustering.hierarchical_clustering import Ward_segment, Ward_quick_segment
import os.path as op
import time
#import fff.parcellation as Pa
import fff2.spatial_models.parcellation as Pa

# ------------------------------------
# 1. Get the data (mask+functional image)
# take several experimental conditions
# time courses could be used instead

nbru = [1]
nbeta = [21,24,25,29,31]
Mask_Images ="/volatile/thirion/Localizer/sujet%02d/functional/fMRI/spm_analysis_RNorm_S/mask.img" % nbru[0]
betas = ["/volatile/thirion/Localizer/sujet%02d/functional/fMRI/spm_analysis_RNorm_S/spmT_%04d.img" % (nbru[0], n) for n in nbeta]
nbparcel = 500


# ------------------------------------
# 2. Read the data
# one mask and several contrast images (or time series images)

import nifti

nim = nifti.NiftiImage(Mask_Images)
ref_dim =  nim.getVolumeExtent()
sform = sform = nim.header['sform']
voxsize = nim.getVoxDims()

mask = np.transpose(nim.getDataArray())
xyz = np.array(np.where(mask>0))
nbvox = np.size(xyz,1)

# from vox to mm
xyz2 = np.hstack((np.transpose(xyz),np.ones((nbvox,1))))
tal = np.dot(xyz2,np.transpose(sform))[:,:3]

Beta = []
for b in range(len(nbeta)):
	rbeta = nifti.NiftiImage(betas[b])
	beta = np.transpose(rbeta.getDataArray())
	beta = beta[mask>0]
	Beta.append(beta)
	
Beta = np.transpose(np.array(Beta))	

# ------------------------------------
#3. Build the 3D model of the data
# remove the small connected components
g = fg.WeightedGraph(nbvox)
# nn=6 yields a quicker solution than nn=18
nn = 6
g.from_3d_grid(np.transpose(xyz.astype(np.int)),nn)
# get the main cc of the graph in order to remove spurious regions
aux = np.zeros(g.V).astype('bool')
imc = g.main_cc()
aux[imc]= True
if np.sum(aux)==0:
	raise ValueError, "empty mask. Cannot proceed"
g = g.subgraph(aux)
mask[xyz[0],xyz[1],xyz[2]]=aux
Beta = Beta[aux,:]
xyz = xyz[:,aux]
nbvox = np.size(xyz,1)
tal = tal[aux,:]

# ------------------------------------
#4. Parcel the data
# 4.a.
# Ward's method : expensive (40 minutes for a 3D image)!
#
mu = 10.0 # weight of anatomical ionformation
feature = np.hstack((Beta,mu*tal/np.std(tal)))

t1 = time.time()
w,cost = Ward_quick_segment(g, feature, stop=-1, qmax=nbparcel)
t2 = time.time()
lpa = Pa.Parcellation(nbparcel,np.transpose(xyz),np.reshape(w,(nbvox,1)))
pi = np.reshape(lpa.population(),nbparcel)
vi = np.sum(lpa.var_feature_intra([Beta])[0],1)
vf = np.dot(pi,vi)/nbvox
va =  np.dot(pi,np.sum(lpa.var_feature_intra([tal])[0],1))/nbvox
print nbparcel, "functional variance", vf, "anatomical variance",va

## 4.b. "random Voronoi" approach: sample with different random seeds
## yields less homogeneous results
##
#V = np.infty
#for i in range(1):
#	from numpy.random import rand
#	seeds = np.argsort(rand(g.V))[:nbparcel]
#	g.set_euclidian(Beta)
#	u = g.Voronoi_Labelling(seeds)
#	lpa = Pa.Parcellation(nbparcel,np.transpose(xyz),np.reshape(u,(nbvox,1)))
#	pi = np.reshape(lpa.population(),nbparcel)
#	vi = np.sum(lpa.var_feature_intra([Beta])[0],1)
#	va =  np.dot(pi,np.sum(lpa.var_feature_intra([tal])[0],1))/nbvox
#	vf = np.dot(pi,vi)/nbvox
#	print  nbparcel, "functional variance", vf, "anatomical variance",va

print t2-t1

# ------------------------------------
#5. write the resulting label image

LabelImage = op.join("/tmp","toto.nii")
Label = -np.ones(ref_dim,'int16')
Label[mask>0] = w
nim = nifti.NiftiImage(np.transpose(Label),rbeta.header)
nim.description='Intra-subject parcellation'
nim.save(LabelImage)
