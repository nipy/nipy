"""
An example to illustrate how to 

* load in a sequence of images
* do some simple calculation (mean + 3 * median) / std
* plot results overlayed on a slice of anatomy

The FIAC data is described here:

Taylor, J.E. & Worsley, K.J. (2005). "Inference for magnitudes 
and delays of responses in the FIAC data using BRAINSTAT/FMRISTAT." 
Human Brain Mapping, 27,434-441.

"""

from numpy import *
import pylab

import os

from neuroimaging.core.image.image import Image

import io

subjects = [0,1,3,4,6,7,8,9,10,11,12,13,14,15]

# load data: sentence block effects from FIAC data

effects = array([Image("%s/FIAC/fixed/block/contrasts/sentence/fiac%d/effect.nii" % (io.prefix, d))[:] for d in subjects])
overall = mean(array([Image("%s/FIAC/fixed/block/contrasts/average/fiac%d/effect.nii" % (io.prefix, d))[:] for d in subjects]), axis=0)

# compute answer

answer = (mean(effects, axis=0) + 3 * median(effects)) / std(effects, axis=0)

# show data

pylab.imshow(mean(effects, axis=0)[40])

# with anatomy

pylab.figure()

# get RGB data

nanswer = pylab.normalize(vmin = answer.min(), vmax = answer.max())
canswer = pylab.cm.hot(nanswer(answer[40]))

anat = Image("%s/FIAC/avg152T1_brain.img" % io.prefix)
nanat = pylab.normalize(vmin = anat[40].min(), vmax = anat[40].max())
canat = pylab.cm.gray(nanat(anat[40]))

mask = greater(answer[40], 3)

rgb = array([canswer[:,:,i] * mask + canat[:,:,i] * (1 - mask) for i in range(3)])
pylab.imshow(transpose(rgb, (1,2,0)))

pylab.contour(overall[40], [0.01])


## a prettier picture with the mean effect instead of the silly
## transformation used above

pylab.figure()
emean = mean(effects, axis=0)

mask = greater(emean[40], 0.005)

nemean = pylab.normalize(vmin = emean.min(), vmax = emean.max())
cemean = pylab.cm.hot(nemean(emean[40]))

rgb = array([cemean[:,:,i] * mask + canat[:,:,i] * (1 - mask) for i in range(3)])
pylab.imshow(transpose(rgb, (1,2,0)))
pylab.contour(overall[40], [0.01])

