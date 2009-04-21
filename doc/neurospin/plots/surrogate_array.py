import numpy as np
import pylab as pl

from nipy.neurospin.utils.simul_2d_multisubject_fmri_dataset import \
     make_surrogate_array

pos = np.array([[10, 10],
                [14, 20],
                [23, 18]])
ampli = np.array([4, 5, 2])

# First generate some noiseless data
noiseless_data = make_surrogate_array(nbsubj=1, noise_level=0, spatial_jitter=0,
                                signal_jitter=0, pos=pos, ampli=ampli)

pl.figure(figsize=(10, 3))
pl.subplot(1, 4, 1)
pl.imshow(noiseless_data[0])
pl.title('Noise-less data')

# Second, generate some group data, with default noise parameters
group_data = make_surrogate_array(nbsubj=3, pos=pos, ampli=ampli)

pl.subplot(1, 4, 2)
pl.imshow(group_data[0])
pl.title('Subject 1')
pl.subplot(1, 4, 3)
pl.title('Subject 2')
pl.imshow(group_data[1])
pl.subplot(1, 4, 4)
pl.title('Subject 3')
pl.imshow(group_data[2])
