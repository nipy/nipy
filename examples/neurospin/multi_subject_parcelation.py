"""
This script contains a quick demo on  a multi'subject parcellation
on a small 2D example
"""
import numpy as np
import nipy.neurospin.spatial_models.hierarchical_parcellation as hp
import nipy.neurospin.utils.simul_multisubject_fmri_dataset as simul
import nipy.neurospin.spatial_models.parcellation as fp

# step 1:  generate some synthetic data
nsubj = 10
dimx = 60
dimy = 60
pos = 3*np.array([[ 6,  7],
                  [10, 10],
                  [15, 10]])
ampli = np.array([5, 7, 6])
sjitter = 6.0
dataset = simul.surrogate_2d_dataset(nbsubj=nsubj, dimx=dimx, dimy=dimy, 
                                     pos=pos, ampli=ampli, width=10.0)
# dataset represents 2D activation images from nsubj subjects,
# with shape (dimx,dimy)

# step 2 : prepare all the information for the parcellation
nbparcel = 10
ref_dim = (dimx,dimy)
xy = np.array(np.where(dataset[0])).T
nvox = np.size(xy,0)
xyz = np.hstack((xy,np.zeros((nvox,1))))
	
ldata = np.reshape(dataset,(nsubj,dimx*dimy,1))
anat_coord = xy
mask = np.ones((nvox,nsubj)).astype('bool')
Pa = fp.Parcellation(nbparcel,xyz,mask-1)

# step 3 : run the algorithm
Pa =  hp.hparcel(Pa, ldata, anat_coord, mu = 3.0)
# note: play with mu to change the 'stiffness of the parcellation'
	
# step 4:  look at the results
Label =  np.array([np.reshape(Pa.label[:,s],(dimx,dimy))
                   for s in range(nsubj)])

import matplotlib.pylab as mp
mp.figure()

for s in range(nsubj):
    mp.subplot(2, 5, s+1)
    mp.imshow(dataset[s], interpolation='nearest')
    mp.axis('off')
mp.figure()

for s in range(nsubj):
    mp.subplot(2, 5, s+1)
    mp.imshow(Label[s], interpolation='nearest', vmin=-1, vmax=nbparcel)
    mp.axis('off')
mp.show()



