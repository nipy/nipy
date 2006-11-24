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

import numpy as N
import scipy.stats
import pylab

from neuroimaging.core.image.image import Image

# Web directory points to an unpacked copy of the rfx dataset described here:
# http://www.fil.ion.ucl.ac.uk/spm/data/multi_sub.html
webdir = 'http://neuroimaging.scipy.org/downloads/rfx_data'

# contrast image numbers in archive (e.g. con_0006.img -> 6)
con_nos = range(6, 18)

# Make nipy image objects from web archive
images = [Image(os.path.join(webdir, 'con_%04d.img' % i)) for i in range(6,18)]

# Create t statistic images
out_images = [Image('t_%04d.nii' % i, mode='w', grid=images[i-6].grid,
                    clobber=True) for i in range(6,18)]

# Make slice iterators for output images
out_iters = [img.slices(mode='w') for img in out_images]

n = len(images)
for i, cur_image in enumerate(images):

    # collect slice iterators for all input images but the current image
    iters = [img.slices() for img in images if img is not cur_image]

    # Iterate over the slices of the current image
    for data in cur_image.slices():
        # Get matching slice from all other images
        out = N.array([it.next() for it in iters])
        # Calculate t value
        mu = out.mean(axis=0)
        std = out.std(axis=0)
        t = (data - mu) / (std * (N.sqrt(1 - 1./n)))
        # Put into matching output image
        out_iters[i].next().set(t)

del(out_images)
out_images = [Image('t_%04d.nii' % i) for i in range(6,18)]

# Plot image t distribution against standard t distribution
for image in out_images:
    print 'Range: ', (N.nanmin(image[:].flat), N.nanmax(image[:].flat))
    imagedata = N.nan_to_num(image[:]).flat
    imagedata = imagedata[N.not_equal(imagedata, 0.)]
    count, x = N.histogram(imagedata, bins=40)
    delta = x[2]-x[1]
    pylab.bar(x, count, width=delta)
    X = N.linspace(-10,10,201)
    tval = scipy.stats.t.cdf(X, n-2)
    tval = (tval[1:] - tval[:-1]) * delta / (X[2] - X[1])
    pylab.plot((X[:-1] + X[1:])/2., tval*N.product(imagedata.shape), 'r',
               linewidth=2)
    pylab.title('Histogram for %s' % image._source.filename)
    pylab.show()
