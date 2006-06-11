import numpy as N
import pylab

from neuroimaging.fmri import fMRIImage 
from neuroimaging.fmri.pca import PCAmontage
from neuroimaging.image import Image

# Load an fMRI image

fmridata = fMRIImage('http://kff.stanford.edu/BrainSTAT/testdata/test_fmri.img')
# Create a mask

frame = fmridata.frame(0)
mask = Image(N.greater(frame.readall(), 500).astype(N.Float), grid=frame.grid)

# Fit PCAmontage which allows you to visualize the results

p = PCAmontage(fmridata, mask=mask)
p.fit()
output = p.images(which=range(4))

# View the results
# compare with "http://www.math.mcgill.ca/keith/fmristat/figs/figpca1.jpg"

p.time_series()
p.montage()
pylab.show()
