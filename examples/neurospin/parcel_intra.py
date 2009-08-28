"""
Example of script to parcellate the data from one subject
And of various processing that can be applied then
author: Bertrand Thirion, 2005-2009
"""


import numpy as np
import os.path as op
import time
import tempfile
from nipy.io.imageformats import load, save, Nifti1Image 
#import nifti

import nipy.neurospin.graph as fg
import nipy.neurospin.graph.field as ff
from nipy.neurospin.clustering.hierarchical_clustering import \
     ward_segment, ward_quick_segment
import nipy.neurospin.spatial_models.parcellation as Pa


import get_data_light
get_data_light.getIt()

# ------------------------------------
# 1. Get the data (mask+functional image)
# take several experimental conditions
# time courses could be used instead

nbeta = [29]
data_dir = op.expanduser(op.join('~', '.nipy', 'tests', 'data'))
MaskImage = op.join(data_dir,'mask.nii.gz')
betas = [op.join(data_dir,'spmT_%04d.nii.gz'%n) for n in nbeta]
swd = tempfile.mkdtemp()

nbparcel = 500

# ------------------------------------
# 2. Read the data
# one mask and several contrast images (or time series images)

#nim = nifti.NiftiImage(MaskImage)
#ref_dim =  nim.getVolumeExtent()
#affine = nim.header['sform']
#mask = nim.asarray().T

nim = load(MaskImage)
ref_dim = nim.get_shape()
affine = nim.get_affine()
mask = nim.get_data()

xyz = np.array(np.where(mask>0))
nvox = np.size(xyz,1)

# from vox to mm
xyza = np.hstack((xyz.T,np.ones((nvox,1))))
coord = np.dot(xyza,affine.T)[:,:3]

beta = []
for b in range(len(nbeta)):
    #rbeta = nifti.NiftiImage(betas[b])
    #lbeta = rbeta.asarray().T
    rbeta = load(betas[b])
    lbeta = rbeta.get_data()
    lbeta = lbeta[mask>0]
    beta.append(lbeta)
    
beta = np.array(beta).T

# ------------------------------------
#3. Build the 3D model of the data
# remove the small connected components
g = fg.WeightedGraph(nvox)
#g = ff.Field(nvox)
# nn=6 yields a quicker solution than nn=18
nn = 6
g.from_3d_grid(xyz.T.astype(np.int),nn)
# get the main cc of the graph in order to remove spurious regions
aux = np.zeros(g.V).astype('bool')
imc = g.main_cc()
aux[imc]= True
if np.sum(aux)==0:
    raise ValueError, "empty mask. Cannot proceed"
g = g.subgraph(aux)
#mask[xyz[0],xyz[1],xyz[2]]=aux
lmask = np.zeros(ref_dim)
lmask[mask>0] = aux

beta = beta[aux,:]
xyz = xyz[:,aux]
nvox = np.size(xyz,1)
coord = coord[aux,:]

# ------------------------------------
#4. Parcel the data
# 4.a.
# Ward's method : expensive (40 minutes for a 3D image)!
#
mu = 10.0 # weight of anatomical ionformation
feature = np.hstack((beta,mu*coord/np.std(coord)))
g = ff.Field(nvox,g.edges,g.weights,feature)

w,J0 = g.ward(nbparcel)
lpa = Pa.Parcellation(nbparcel,xyz.T,np.reshape(w,(nvox,1)))
pi = np.reshape(lpa.population(),nbparcel)
vi = np.sum(lpa.var_feature_intra([beta])[0],1)
vf = np.dot(pi,vi)/nvox
va =  np.dot(pi,np.sum(lpa.var_feature_intra([coord])[0],1))/nvox
print nbparcel, "functional variance", vf, "anatomical variance",va


## 4.b. "random Voronoi" approach: sample with different random seeds
#from numpy.random import rand
#seeds = np.argsort(rand(g.V))[:nbparcel]
#seeds, u, J1 = g.geodesic_kmeans(seeds)
seeds, u, J1 = g.geodesic_kmeans(label=w)
lpa = Pa.Parcellation(nbparcel,xyz.T,np.reshape(u,(nvox,1)))
pi = np.reshape(lpa.population(),nbparcel)
vi = np.sum(lpa.var_feature_intra([beta])[0],1)
va =  np.dot(pi,np.sum(lpa.var_feature_intra([coord])[0],1))/nvox
vf = np.dot(pi,vi)/nvox
print  nbparcel, "functional variance", vf, "anatomical variance",va

## yields less homogeneous results
##
#V = np.infty
#for i in range(1):
#   from numpy.random import rand
#   seeds = np.argsort(rand(g.V))[:nbparcel]
#   g.set_euclidian(beta)
#   u = g.Voronoi_Labelling(seeds)
#   lpa = Pa.Parcellation(nbparcel,xyz.T,np.reshape(u,(nvox,1)))
#   pi = np.reshape(lpa.population(),nbparcel)
#   vi = np.sum(lpa.var_feature_intra([beta])[0],1)
#   va =  np.dot(pi,np.sum(lpa.var_feature_intra([coord])[0],1))/nvox
#   vf = np.dot(pi,vi)/nvox
#   print  nbparcel, "functional variance", vf, "anatomical variance",va


# ------------------------------------
#5. write the resulting label image

LabelImage = op.join(swd,"parcel_wards.nii")
Label = -np.ones(ref_dim,'int16')
Label[lmask>0] = w
#nim = nifti.NiftiImage(Label.T,rbeta.header)
#nim.description='Intra-subject parcellation'
#nim.save(LabelImage)
wim = Nifti1Image (Label, affine)
hdr = wim.get_header()
hdr['descrip']='Intra-subject parcellation'
save(wim, LabelImage)

LabelImage = op.join(swd,"parcel_gkmeans.nii")
Label = -np.ones(ref_dim,'int16')
Label[lmask>0] = u
wim = Nifti1Image (Label, affine)
hdr = wim.get_header()
hdr['descrip']='Intra-subject parcellation'
save(wim, LabelImage)
#nim = nifti.NiftiImage(Label.T,rbeta.header)
#nim.description='Intra-subject parcellation'
#nim.save(LabelImage)

print "Wrote two parcel images as %s and %s" %\
      (op.join(swd,"parcel_wards.nii"),op.join(swd,"parcel_gkmeans.nii"))
