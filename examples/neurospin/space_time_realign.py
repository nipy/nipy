from neuroimaging.neurospin import register 
from neuroimaging.neurospin import neuro

import numpy as np
from os.path import join
import sys
import time

rootpath = 'D:\\data\\fiac'
runnames = ['run1', 'run1']

im1 = neuro.image(join(rootpath, runnames[0]+'.nii'))
im2 = neuro.image(join(rootpath, runnames[1]+'.nii'))

# Test hack
dat1 = im1.array[:,:,:,0:5]
dat2 = im2.array[:,:,:,5:10]

run1 = register.TimeSeries(dat1, toworld=im1.transform, tr=2.5, 
                           slice_order='ascending', interleaved=True)
run2 = register.TimeSeries(dat2, toworld=im2.transform, tr=2.5, 
                           slice_order='ascending', interleaved=True)

"""
transforms = register.realign4d(run1, within_loops=1) 
corr_run1 = register.resample4d(run1, transforms=transforms)
"""

transforms = register.realign4d([run1, run2], within_loops=0) 
corr_run1 = register.resample4d(run1, transforms=transforms[0])
corr_run2 = register.resample4d(run2, transforms=transforms[1])

