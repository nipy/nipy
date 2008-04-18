#!/bin/env python
'''
Example to show use of some common image generators
'''

import numpy as np

import neuroimaging.core.image.generators as g 
from neuroimaging.core.api import Image, SamplingGrid

def _main():
    img = Image(np.zeros((3, 4, 5)),
                SamplingGrid.from_start_step(['z', 'y', 'x'],
                                             (0,)*3,
                                             (1,)*3,
                                             (3,4,5)))

    # Slice along the 0th axis (default)
    print "slicing along axis=0"
    for i, s in g.data_generator(img):
        print s, s.shape

    # Slice along the 1st axis
    print "slicing along axis=1"
    for i, s in g.slice_generator(img, axis=1):
        print s, s.shape

    # Slice along the 2nd axis, writing the z index
    # to each point in the image
    print "Writing z index values"

    z = 0
    for i, s in g.slice_generator(img, axis=2):
        s[:] = z
        z += 1

    print np.asarray(img)
    
    # copy img into B

    B = Image(np.ones((3, 4, 5)),
              SamplingGrid.from_start_step(['z', 'y', 'x'],
                                           (0,)*3,
                                           (1,)*3,
                                           (3,4,5)))

    for i, s in g.slice_generator(img):
        B[i] = s

    # copy img into B, swapping the 0th and 1st axes
    B = Image(np.zeros((4, 3, 5)),
              SamplingGrid.from_start_step(['z', 'y', 'x'],
                                           (0,)*3,
                                           (1,)*3,
                                           (4,3,5)))

    # copy img into B, swapping the 1st and 2nd axes

    for i, s in g.slice_generator(img, axis=2):
        B[i] = s.T

    print np.asarray(B), B.shape

    for i, s in g.data_generator(img, iterable=[2,0,1]):
        print s
    
    print "========"

    parcelmap = np.asarray([[0,0,1,2],[0,1,1,2],[0,0,0,2]])
    parcelseq = ((1, 2), (0,), (2,))

    for i, d in g.data_generator(img, g.parcels(parcelmap,labels=parcelseq)):
        print d.shape


if __name__ == '__main__':
    _main()
