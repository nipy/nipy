#!/usr/bin/env python 

import fff2.registration as registration
import fff2.utils.io as io
from os.path import join
import numpy as np


"""
Example of running affine matching on the 'sulcal2000' database
"""

##rootpath = 'D:\\data\\sulcal2000'
rootpath = '/neurospin/lnao/Panabase/roche/sulcal2000'
        
print('Scanning data directory...')

# Get data
print('Fetching image data...')
I = io.image()
J = io.image()
I.load(join(rootpath,'nobias_anubis'+'.nii'))
J.load(join(rootpath,'ammon_TO_anubis'+'.nii'))

# Setup registration algorithm
matcher = registration.iconic(I, J) ## I: source, J: target

# Params
size = 5
nsimu = 1
depth = 10

import pylab as pl

# Simulations 
for i in range(nsimu): 

    # Select random block
    x0 = np.random.randint(I.array.shape[0]-size)
    y0 = np.random.randint(I.array.shape[1]-size)
    z0 = np.random.randint(I.array.shape[2]-size)
    matcher.set(corner=[x0,y0,z0], size=[size,size,size])

    # Explore neighborhood
    tx = I.voxsize[0] * (np.arange(2*depth + 1)-depth)
    s, p = matcher.explore(tx=tx)

    # Display
    pl.plot(p[:,0],(1-s)**(-.5*size**3)) 



pl.show()

