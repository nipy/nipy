#!/usr/bin/env python 

from neuroimaging.neurospin import register 

# Use Matthew's volumeimages for I/O. 
import volumeimages

from os.path import join
import sys
import time


"""
Example of running affine matching on the 'sulcal2000' database
"""

# Dirty hack for me to be able to access data from my XP environment
rootpath = '/neurospin/lnao/Panabase/roche/sulcal2000'
from os import name
if name == 'nt':
	rootpath = 'D:\\data\\sulcal2000'

        
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
I = volumeimages.load(join(rootpath,'nobias_'+source+'.nii'))
J = volumeimages.load(join(rootpath,'nobias_'+target+'.nii'))

# Perform affine normalization 
print('Setting up registration...')
tic = time.time()
T = register.iconic_matching(I.get_data(), J.get_data(), 
			     I.get_affine(), J.get_affine(), 
			     similarity=similarity, 
			     interp=interp, 
			     normalize=normalize, 
			     optimizer=optimizer)
toc = time.time()
print('  Registration time: %f sec' % (toc-tic))

# Resample source image
print('Resampling source image...')
tic = time.time()
It_data = register.transform.resample(T, I.get_data(), J.get_data(), I.get_affine(), J.get_affine())
It = volumeimages.nifti1.Nifti1Image(affine=J.get_affine(), data=It_data)
toc = time.time()
print('  Resampling time: %f sec' % (toc-tic))


# Save resampled source
outfile =  source+'_TO_'+target+'.nii'
print ('Saving resampled source in: %s' % outfile)
It.to_files(outfile)

# Save transformation matrix
"""
import numpy as np
np.save(outfile, T)
"""
