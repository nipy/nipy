import fff2
import numpy as np

# Load an image 
imfile = '/volatile/roche/database/fiac/sub0/fMRI/acquisition/run1/run1.nii'
imI = fff2.neuro.image(imfile)

"""
I = np.random.rand(20,20,15,100)
imI = fff2.neuro.image(I)
"""

# Create a fMRI image object to handle time parameters
imI = fff2.neuro.fmri_image(imI, tr=2.5, interleaved=True)

# Correct for slice timing 
imIs, params = fff2.neuro.fmri.realign4d(imI, motion_loops=3, speedup=4, optimizer='powell')

# Plot
"""
from pylab import *
plot(params[:,0:3])
show()
"""

# Save corrected image 
imIs.save('toto.nii')
