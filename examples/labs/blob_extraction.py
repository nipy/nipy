# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This scipt generates a noisy activation image image
and extracts the blobs from it.

Author : Bertrand Thirion, 2009
"""
#autoindent
print __doc__

import numpy as np
import pylab as pl
import matplotlib

import nipy.labs.utils.simul_multisubject_fmri_dataset as simul
import nipy.labs.spatial_models.hroi as hroi
from nipy.labs.spatial_models.discrete_domain import domain_from_array

# ---------------------------------------------------------
# simulate an activation image
# ---------------------------------------------------------

dimx = 60
dimy = 60
pos = np.array([[12, 14], [20, 20], [30, 20]])
ampli = np.array([3, 4, 4])

nbvox = dimx * dimy
dataset = simul.surrogate_2d_dataset(nbsubj=1, dimx=dimx, dimy=dimy, pos=pos,
                                     ampli=ampli, width=10.0).squeeze()
values = dataset.ravel()


#-------------------------------------------------------
# Computations
#------------------------------------------------------- 

# create a domain descriptor associated with this
domain = domain_from_array(dataset ** 2 > 0)

nroi = hroi.HROI_as_discrete_domain_blobs(domain, dataset.ravel(),
                                          threshold=2.0, smin=3)
label = np.reshape(nroi.label, ((dimx, dimy)))

# create an average activaion image
nroi.make_feature('activation', dataset.ravel())
mean_activation = nroi.representative_feature('activation')
bmap = - np.ones((dimx, dimy))
for k in range(nroi.k):
    bmap[label == k] = mean_activation[k]

#--------------------------------------------------------
# Result display
#--------------------------------------------------------

aux1 = (0 - values.min()) / (values.max() - values.min())
aux2 = (bmap.max() - values.min()) / (values.max() - values.min())
cdict = {'red': ((0.0, 0.0, 0.7), 
                 (aux1, 0.7, 0.7),
                 (aux2, 1.0, 1.0),
                 (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 0.7),
                   (aux1, 0.7, 0.0),
                   (aux2, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),
         'blue': ((0.0, 0.0, 0.7),
                  (aux1, 0.7, 0.0),
                  (aux2, 0.5, 0.5),
                  (1.0, 1.0, 1.0))}
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)

pl.figure(figsize=(12, 3))
pl.subplot(1, 3, 1)
pl.imshow(dataset, interpolation='nearest', cmap=my_cmap)
cb = pl.colorbar()
for t in cb.ax.get_yticklabels():
    t.set_fontsize(16)
    
pl.axis('off')
pl.title('Thresholded data')

# plot the blob label image
pl.subplot(1, 3, 2)
pl.imshow(label, interpolation='nearest')
pl.colorbar()
pl.title('Blob labels')

# plot the blob-averaged signal image
aux = 0.01
cdict = {'red': ((0.0, 0.0, 0.7), (aux, 0.7, 0.7), (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 0.7), (aux, 0.7, 0.0), (1.0, 1.0, 1.0)),
         'blue': ((0.0, 0.0, 0.7), (aux, 0.7, 0.0), (1.0, 0.5, 1.0))}
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)

pl.subplot(1, 3, 3)
pl.imshow(bmap, interpolation='nearest', cmap=my_cmap)
cb = pl.colorbar()
for t in cb.ax.get_yticklabels():
    t.set_fontsize(16)
pl.axis('off')
pl.title('Blob average')
pl.show()
