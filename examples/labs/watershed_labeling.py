# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This scipt generates a noisy activation image image
and performs a watershed segmentation in it.

Author : Bertrand Thirion, 2009--2012
"""
#autoindent
print __doc__

import numpy as np
import matplotlib
import matplotlib.pylab as mp

from nipy.labs.spatial_models.hroi import HROI_from_watershed
from nipy.labs.spatial_models.discrete_domain import grid_domain_from_shape
import nipy.labs.utils.simul_multisubject_fmri_dataset as simul

###############################################################################
# data simulation

shape = (60, 60)
pos = np.array([[12, 14],
                [20, 20],
                [30, 20]])
ampli = np.array([3, 4, 4])
x = simul.surrogate_2d_dataset(n_subj=1, shape=shape, pos=pos, ampli=ampli, 
                               width=10.0).squeeze()

th = 2.36

# compute the field structure and perform the watershed
domain = grid_domain_from_shape(shape)
nroi = HROI_from_watershed(domain, np.ravel(x), threshold=th)
label = nroi.label

#compute the region-based signal average
bfm = np.array([np.mean(x.ravel()[label == k]) for k in range(label.max() + 1)])
bmap = np.zeros(x.size)
if label.max() > - 1:
    bmap[label > - 1] = bfm[label[label > - 1]]

label = np.reshape(label, shape)
bmap = np.reshape(bmap, shape)

###############################################################################
# plot the input image

aux1 = (0 - x.min()) / (x.max() - x.min())
aux2 = (bmap.max() - x.min()) / (x.max() - x.min())
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

mp.figure(figsize=(12, 3))
mp.subplot(1, 3, 1)
mp.imshow(np.squeeze(x), interpolation='nearest', cmap=my_cmap)
mp.axis('off')
mp.title('Thresholded image')

cb = mp.colorbar()
for t in cb.ax.get_yticklabels():
    t.set_fontsize(16)

###############################################################################
# plot the watershed label image
mp.subplot(1, 3, 2)
mp.imshow(label, interpolation='nearest')
mp.axis('off')
mp.colorbar()
mp.title('Labels')

###############################################################################
# plot the watershed-average image
mp.subplot(1, 3, 3)
aux = 0.01
cdict = {'red': ((0.0, 0.0, 0.7), (aux, 0.7, 0.7), (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 0.7), (aux, 0.7, 0.0), (1.0, 1.0, 1.0)),
         'blue': ((0.0, 0.0, 0.7), (aux, 0.7, 0.0), (1.0, 0.5, 1.0))}
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)

mp.imshow(bmap, interpolation='nearest', cmap=my_cmap)
mp.axis('off')
mp.title('Label-average')

cb = mp.colorbar()
for t in cb.ax.get_yticklabels():
    t.set_fontsize(16)

mp.show()
