#!/usr/bin/env python 

"""
Example of running inter-subject affine matching on the sulcal2000
database acquired at SHFJ, Orsay, France. 
"""

from neuroimaging.neurospin.affine_register import register, resample

# Use Matthew's volumeimages for I/O. 
import volumeimages as v

from os.path import join
import sys
import time

rootpath = '/neurospin/lnao/Panabase/roche/sulcal2000'
# Unimportant hack...
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

# Print messages 
print ('Source brain: %s' % source)
print ('Target brain: %s' % target)
print ('Similarity measure: %s' % similarity)
print ('Optimizer: %s' % optimizer)

# Get data
print('Fetching image data...')
I = v.load(join(rootpath,'nobias_'+source+'.nii'))
J = v.load(join(rootpath,'nobias_'+target+'.nii'))

# Perform affine normalization 
print('Setting up registration...')
tic = time.time()
T = register(I, J, similarity=similarity, interp=interp, normalize=normalize, optimizer=optimizer)
toc = time.time()
print('  Registration time: %f sec' % (toc-tic))


# Resample source image
print('Resampling source image...')
tic = time.time()
It =  v.nifti1.Nifti1Image(affine=J.get_affine(), data=resample(I, J, T))
toc = time.time()
print('  Resampling time: %f sec' % (toc-tic))


# Save resampled source
outfile =  source+'_TO_'+target+'.nii'
print ('Saving resampled source in: %s' % outfile)
v.save(It, outfile)


# Save transformation matrix
"""
import numpy as np
np.save(outfile, T)
"""
