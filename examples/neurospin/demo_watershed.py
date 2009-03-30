"""
This scipt generates a noisy activation image image
and performs a watershed segmentation in it.

Author : Bertrand Thirion, 2009
"""
#autoindent

import numpy as np
import fff2.graph.field as ff
import fff2.utils.simul_2d_multisubject_fmri_dataset as simul
import matplotlib
import matplotlib.pylab as mp

################################################################################
# data simulation
dimx=60
dimy=60
pos = 2*np.array([[ 6,  7],
                  [10, 10],
                  [15, 10]])
ampli = np.array([3, 4, 4])

nbvox = dimx*dimy
x = simul.make_surrogate_array(nbsubj=1, dimx=dimx, dimy=dimy,
                               pos=pos, ampli=ampli, width=10.0)

x = np.reshape(x, (dimx, dimy, 1))
beta = np.reshape(x, (nbvox, 1))

xyz = np.array(np.where(x))
nbvox = np.size(xyz,1)
th = 2.36

# compute the field structure and perform the watershed
Fbeta = ff.Field(nbvox)
Fbeta.from_3d_grid(xyz.T.astype('i'), 18)
Fbeta.set_field(beta)
idx, depth, major, label = Fbeta.custom_watershed(0,th)

#compute the region-based signal average
bfm = np.array([np.mean(beta[label==k]) for k in range(label.max()+1)])
bmap = np.zeros(nbvox)
if label.max()>-1:
    bmap[label>-1]= bfm[label[label>-1]]

label = np.reshape(label, (dimx, dimy))
bmap  = np.reshape(bmap,  (dimx, dimy))

################################################################################
# plot the input image
aux1 = (0-x.min())/(x.max()-x.min())
aux2 = (bmap.max()-x.min())/(x.max()-x.min())
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
                 (1.0, 1.0, 1.0))
        }
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)

mp.figure(figsize=(12, 3))
mp.subplot(1, 3, 1)
mp.imshow(np.squeeze(x), interpolation='nearest', cmap=my_cmap)
mp.axis('off')
mp.title('Thresholded image')

cb = mp.colorbar()
for t in cb.ax.get_yticklabels():
     t.set_fontsize(16)

################################################################################
# plot the watershed label image
mp.subplot(1, 3, 2)
mp.imshow(label, interpolation='nearest')
mp.axis('off')
mp.colorbar()
mp.title('Labels')

################################################################################
# plot the watershed-average image
mp.subplot(1, 3, 3)
aux = 0.01#(th-bmap.min())/(bmap.max()-bmap.min())
cdict = {'red': ((0.0, 0.0, 0.7), (aux, 0.7, 0.7),(1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 0.7), (aux, 0.7, 0.0),(1.0, 1.0, 1.0)),
         'blue': ((0.0, 0.0, 0.7), (aux, 0.7, 0.0),(1.0, 0.5, 1.0))}
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)

mp.imshow(bmap, interpolation='nearest', cmap=my_cmap)
mp.axis('off')
mp.title('Label-average')

cb = mp.colorbar()
for t in cb.ax.get_yticklabels():
     t.set_fontsize(16)

mp.show()
