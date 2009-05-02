"""
This scipt generates a noisy activation image image
and extracts the blob from it.

Author : Bertrand Thirion, 2009
"""
#autoindent

import numpy as np
import fff2.graph.field as ff
import fff2.utils.simul_2d_multisubject_fmri_dataset as simul


dimx=60
dimy=60
pos = 2*np.array([[6,7],[10,10],[15,10]])
ampli = np.array([3,4,4])

nbvox = dimx*dimy
dataset = simul.make_surrogate_array(nbsubj=1,dimx=dimx,dimy=dimy,pos=pos,ampli=ampli,width=10.0,fid='dataset.dat')
dataset = np.fromfile('dataset.dat')
x = np.reshape(dataset,(dimx,dimy,1))
beta = np.reshape(x,(nbvox,1))

xyz = np.array(np.where(x))
nbvox = np.size(xyz,1)

# build the field
F = ff.Field(nbvox)
F.from_3d_grid(xyz.T,18)
F.set_field(beta)

# compute the blobs
th = 2.36
smin = 5
nroi = F.generate_blobs(refdim=0,th=th,smin = smin)

# compute the average signal within each blob
nroi = nroi.reduce_to_leaves()
idx = nroi.get_seed()
parent = nroi.get_parent()
label = nroi.get_label()
nroi.make_feature(beta, 'height','mean')
bfm = nroi.get_ROI_feature('height')

# plot the input image
bmap = np.zeros(nbvox)
if nroi.k>0:
    bmap[label>-1]= bfm[label[label>-1]]

label = np.reshape(label,(dimx,dimy))
bmap = np.reshape(bmap,(dimx,dimy))

from pylab import *
aux1 = (0-x.min())/(x.max()-x.min())
aux2 = (bmap.max()-x.min())/(x.max()-x.min())
cdict = {'red': ((0.0, 0.0, 0.7), (aux1, 0.7, 0.7),(aux2, 1.0, 1.0),(1.0, 1.0, 1.0)),
       'green': ((0.0, 0.0, 0.7), (aux1, 0.7, 0.0),(aux2, 1.0, 1.0),(1.0, 1.0, 1.0)),
        'blue': ((0.0, 0.0, 0.7), (aux1, 0.7, 0.0),(aux2, 0.5, 0.5),(1.0, 1.0, 1.0))}
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)


import matplotlib.pylab as mp
mp.figure()
mp.imshow(np.squeeze(x),interpolation='nearest',cmap=my_cmap)
cb = mp.colorbar()
for t in cb.ax.get_yticklabels():
     t.set_fontsize(16)
mp.axis('off')
mp.show()

# plot the blob label image
mp.figure()
mp.imshow(label,interpolation='nearest')
mp.colorbar()
mp.show()

# plot the blob-avergaed signal image
mp.figure()
aux = 0.01#(th-bmap.min())/(bmap.max()-bmap.min())
cdict = {'red': ((0.0, 0.0, 0.7), (aux, 0.7, 0.7),(1.0, 1.0, 1.0)),
       'green': ((0.0, 0.0, 0.7), (aux, 0.7, 0.0),(1.0, 1.0, 1.0)),
        'blue': ((0.0, 0.0, 0.7), (aux, 0.7, 0.0),(1.0, 0.5, 1.0))}
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)

mp.imshow(bmap,interpolation='nearest',cmap=my_cmap)
cb = mp.colorbar()
for t in cb.ax.get_yticklabels():
     t.set_fontsize(16)
mp.axis('off')
mp.show()
