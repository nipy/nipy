#!/bin/env python
'''
Extract the means from different parcels within a functional dataset;
show use of iterators in image to do this.  Show use of region
iterators

Usage:
parcel_means.py
'''

import numpy as N
import pylab

from neuroimaging.modalities.fmri import fMRIImage
from neuroimaging.utils.tests.data import repository
from neuroimaging.core.image.iterators import fMRIParcelIterator

subject_no=0
run_no = 1
func_img = fMRIImage('FIAC/fiac%d/fonc%d/fsl/filtered_func_data.img' %
                     (subject_no, run_no),
                     datasource=repository)

# Create array of same shape as one time point of functional image
parcel_arr = N.zeros(func_img.grid.shape[1:])

# Create 2 pretend parcels in array at middle, and off-centre
offset=5
middle_block_def = [slice(i/2-offset, i/2+offset, 1) for i in parcel_arr.shape]
parcel_arr[middle_block_def] = 1
off_block_def = [slice(i/2-2*offset, i/2-offset, 1) for i in parcel_arr.shape]
parcel_arr[off_block_def] = 2

# Set up parcel iteration for functional image
it = fMRIParcelIterator(func_img, parcel_arr[:], [1, 2])

# Iterate to collect means over parcels in functional image
means = {}
parcel_labels = []
for d in it:
    L = it.item.label
    means[L] = N.mean(d, axis=1)
    parcel_labels.append(L)
    
# Now iterate over regions
# changing the parcelseq to select "regions"
it = fMRIParcelIterator(func_img, parcel_arr[:], [0, 2])
for d in it:
    print d.shape

# Show the figure
pylab.plot(means[(1,)], means[(2,)], 'bo')
pylab.title('Parcel means')
pylab.xlabel('Mean signal - parcel %s' % parcel_labels[0])
pylab.ylabel('Mean signal - parcel %s' % parcel_labels[1])
pylab.show()
