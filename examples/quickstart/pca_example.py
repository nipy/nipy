import numpy as N
import pylab

from neuroimaging.modalities.fmri import fMRIImage 
from neuroimaging.modalities.fmri.pca import PCAmontage
from neuroimaging.core.image import Image

# Load an fMRI image

fmridata = fMRIImage("test_fmri.hdr", datasource=repository)
# Create a mask

frame = fmridata.frame(0)
mask = Image(N.greater(frame[:], 500).astype(N.float64), grid=frame.grid)

# Fit PCAmontage which allows you to visualize the results

p = PCAmontage(fmridata, mask=mask)
p.fit()
output = p.images(which=range(4))

# View the results
# compare with "http://www.math.mcgill.ca/keith/fmristat/figs/figpca1.jpg"

p.time_series()
p.montage()
pylab.show()
