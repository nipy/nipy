#!/bin/env python
'''
Example to show use of slice and slice/parcel iterators
'''

import numpy as N

from neuroimaging.core.api import Image

def _main():
    img = Image(N.zeros((3, 4, 5)))

    # Slice along the 0th axis (default)
    print "slicing along axis=0"
    for s in img.slice_iterator():
        print s, s.shape

    # Slice along the 1st axis
    print "slicing along axis=1"
    for s in img.slice_iterator(axis=1):
        print s, s.shape

    # Slice along the 2nd axis, writing the z index
    # to each point in the image
    print "Writing z index values"
    z = 0
    for s in img.slice_iterator(axis=2, mode='w'):
        s.set(z)
        z += 1

    print img[:]

    # copy img into B
    B = Image(N.zeros((3, 4, 5)))
    B.from_slice_iterator(img.slice_iterator())
    print B[:]

    # copy img into B, swapping the 0th and 1st axes
    B = Image(N.zeros((4, 3, 5)))
    B.from_slice_iterator(img.slice_iterator(axis=1))
    print B[:]

    # copy img into B, swapping the 1st and 2nd axes
    B = Image(N.zeros((3, 5, 4)))
    B.from_slice_iterator(img.slice_iterator(axis=2), axis=1)
    print B[:]
    
    # use an arbitrary iterator for the source and destination
    B.from_iterator(Image.SliceIterator(img, axis=2),
                    Image.SliceIterator(None, axis=1))
    print B[:]
    

    x = 0
    for s in B.slice_iterator(mode='w'):
        s.set(x)
        x += 1

    print B[:]
    print "========"

    parcelmap = N.asarray([[0,0,0,1,2],[0,0,1,1,2],[0,0,0,0,2]])
    parcelseq = ((1, 2), (0,), (2,))
    i = B.slice_parcel_iterator(parcelmap, parcelseq)
    for n in i:
        print n


    y = 0
    for s in B.slice_iterator(axis=1, mode='w'):
        s.set(y)
        y += 1

    i = B.slice_parcel_iterator(parcelmap, parcelseq)
    for n in i:
        print n

    for s in img.slice_iterator(axis=[0,1]):
        print s

    for s in img.slice_iterator(axis=[0,2]):
        print s

    for s in img.slice_iterator(axis=[1,2]):
        print s


if __name__ == '__main__':
    _main()
