#!/bin/env python
"""
An example application to take a group of images and
compute the (one-sample) studentized residuals from them.

For clarity, this example computes the studentized residuals
the slow way, i.e. recomputing variance 12 times.

You will need write permission in the working directory for this
example.
"""

import os
import copy

import numpy as np
import scipy.stats
import pylab

from neuroimaging.core.api import Image, load_image

# Web directory points to an unpacked copy of the rfx dataset described here:
# http://www.fil.ion.ucl.ac.uk/spm/data/multi_sub.html
webdir = 'http://neuroimaging.scipy.org/downloads/rfx_data'

# contrast image numbers in archive (e.g. con_0006.img -> 6)
con_nos = range(6, 18)
n_cons = len(con_nos)

# Make nipy image objects from web archive
images = [load_image(os.path.join(webdir, 'con_%04d.img' % i)) for i in con_nos]

# Create t statistic images
zeroimg = Image(np.zeros(images[0].grid.shape), images[0].grid)
out_images = [save(zeroimg, 't_%04d.nii' % i, mode='w', 
                   clobber=True) for i in con_nos]

# Make slice iterators for output images
out_iters = [slice_iterator(img, mode='w') for img in out_images]

for i, cur_image in enumerate(images):

    # collect slice iterators for all input images but the current image
    iters = [slice_iterator(img) for img in images if img is not cur_image]

    # Iterate over the slices of the current image
    for data in cur_image.slice_iterator():
        # Get matching slice from all other images
        out = np.array([it.next() for it in iters])
        # Calculate t value
        mu = out.mean(axis=0)
        std = out.std(axis=0)
        t = (data - mu) / (std * (np.sqrt(1 - 1./n_cons)))
        # Put into matching output image
        out_iters[i].next().set(t)

del(out_images)
out_images = [load_image('t_%04d.nii' % i) for i in con_nos]

# Plot image value histogram against standard t distribution
n_cols = 2
n_rows = np.ceil(n_cons / n_cols)
signal_range = [-7.5, 7.5]
count_range = [0, 12000]
    
for img_no, image in enumerate(out_images):
    pylab.subplot(n_rows, n_cols, img_no+1)

    print 'Range: ', (np.nanmin(image[:].flat), np.nanmax(image[:].flat))

    # Plot the image histogram
    imagedata = np.nan_to_num(image[:]).flat
    imagedata = imagedata[np.not_equal(imagedata, 0.)]
    count, x = np.histogram(imagedata, bins=40)
    delta = x[2]-x[1]
    pylab.bar(x, count, width=delta)

    # Plot the t distribution
    X = np.linspace(-10,10,201)
    tval = scipy.stats.t.cdf(X, n_cons-2)
    tval = (tval[1:] - tval[:-1]) * delta / (X[2] - X[1])
    pylab.plot((X[:-1] + X[1:])/2., tval*np.product(imagedata.shape), 'r',
               linewidth=2)

    axes = pylab.gca()
    axes.set_xlim(signal_range)
    axes.set_ylim(count_range)

    pylab.title('Histogram for %s' % image._source.filename)

# Show the figure
pylab.show()
