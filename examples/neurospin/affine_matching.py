#!/usr/bin/env python 

import fff2

from os.path import join
import sys
import time


"""
Example of running affine matching on the 'sulcal2000' database
"""

# Dirty hack for me to be able to access data from my XP environment
from os import name
if name == 'nt':
	rootpath = 'D:\\data\\sulcal2000'
else:
	rootpath = '/neurospin/lnao/Panabase/roche/sulcal2000'
        
print('Scanning data directory...')
source = sys.argv[1]
target = sys.argv[2]
similarity = 'correlation ratio'
if len(sys.argv)>3: 
	similarity = sys.argv[3]
interp = 'partial volume'
if len(sys.argv)>4: 
	interp = sys.argv[4]
normalize = None
if len(sys.argv)>5: 
	normalize = sys.argv[5]
optimizer = 'powell'
if len(sys.argv)>6: 
	optimizer = sys.argv[6]

# Change this to use another I/O package
iolib = 'pynifti'

## Info
print ('Source brain: %s' % source)
print ('Target brain: %s' % target)
print ('Similarity measure: %s' % similarity)
print ('Optimizer: %s' % optimizer)

# Get data
print('Fetching image data...')
I = fff2.neuro.image(join(rootpath,'nobias_'+source+'.nii'), iolib=iolib)
J = fff2.neuro.image(join(rootpath,'nobias_'+target+'.nii'), iolib=iolib)

# Perform affine normalization 
T, It = fff2.neuro.affine_registration(I, J, similarity=similarity, interp=interp, 
				       normalize=normalize, optimizer=optimizer, resample=True)


# Save resampled source
outfile =  source+'_TO_'+target+'.nii'
print ('Saving resampled source in: %s' % outfile)
It.save(outfile)

# Save transformation matrix
import numpy as np
np.save(outfile, T)

