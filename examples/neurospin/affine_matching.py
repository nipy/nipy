#!/usr/bin/env python 

from neuroimaging.neurospin import register 
from neuroimaging.neurospin import neuro


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
similarity = 'cr'
if len(sys.argv)>3: 
	similarity = sys.argv[3]
interp = 'pv'
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
I = neuro.image(join(rootpath,'nobias_'+source+'.nii'), iolib=iolib)
J = neuro.image(join(rootpath,'nobias_'+target+'.nii'), iolib=iolib)

# Perform affine normalization 
print('Setting up registration...')
tic = time.time()
T = register.imatch(I.array, J.array, I.transform, J.transform, 
		    similarity=similarity, 
		    interp=interp, 
		    normalize=normalize, 
		    optimizer=optimizer)
toc = time.time()
print('  Registration time: %f sec' % (toc-tic))

# Resample source image
print('Resampling source image...')
tic = time.time()
It = neuro.image(J)
It.set_array(register.transform.resample(T, I.array, J.array, I.transform, J.transform))
toc = time.time()
print('  Resampling time: %f sec' % (toc-tic))


# Save resampled source
outfile =  source+'_TO_'+target+'.nii'
print ('Saving resampled source in: %s' % outfile)
It.save(outfile)

# Save transformation matrix
"""
import numpy as np
np.save(outfile, T)
"""
