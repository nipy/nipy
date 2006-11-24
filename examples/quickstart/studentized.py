"""
An example application to take a group of images and
compute the (one-sample) studentized residuals from them.

For clarity, this example computes the studentized residuals
the slow way, i.e. recomputing variance 12 times.

"""

import os, copy

import numpy as N
import scipy.stats
import pylab

from neuroimaging.core.image.image import Image

webdir = 'http://kff.stanford.edu/nipy/rfx-data/'
images = [Image(os.path.join(webdir, 'con_%04d.img' % i)) for i in range(6,18)]

out_images = [Image('t_%04d.nii' % i, mode='w', grid=images[i-6].grid,
                    clobber=True) for i in range(6,18)]

out_iters = [img.slices(mode='w') for img in out_images]

n = len(images)
for i, cur_image in enumerate(images):

    iters = [img.slices() for img in images if img is not cur_image]
    for data in cur_image.slices():
        out = N.array([it.next() for it in iters])
        mu = out.mean(axis=0)
        std = out.std(axis=0)
        t = (data - mu) / (std * (N.sqrt(1 - 1./n)))
        out_iters[i].next().set(t)

del(out_images)
out_images = [Image('t_%04d.nii' % i) for i in range(6,18)]


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
