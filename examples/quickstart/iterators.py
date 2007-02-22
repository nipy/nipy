#!/bin/env python
'''
Example to show use of slice and slice/parcel iterators
'''

import numpy as N

from neuroimaging.core.image.iterators import SliceIterator, SliceParcelIterator
from neuroimaging.core.api import Image

def _main():
    img = Image(N.zeros((3, 4, 5)))

    # Slice along the 0th axis (default)
    print "slicing along axis=0"
    for s in SliceIterator(img):
        print s, s.shape


    # Slice along the 1st axis
    print "slicing along axis=1"
    for s in SliceIterator(img, axis=1):
        print s, s.shape

    # Slice along the 2nd axis, writing the z index
    # to each point in the image
    print "Writing z index values"
    z = 0
    for s in SliceIterator(img, axis=2, mode='w'):
        s.set(z)
        z += 1

    print img[:]

    # Slice using the image method interface
    print "slicing with .slice_iterator() method"
    for s in img.slice_iterator():
        print s, s.shape

    print "...and along axis=1"
    for s in img.slice_iterator(axis=1):
        print s, s.shape

    print "...and writing along the y axis"
    y = 0
    for s in img.slice_iterator(mode='w', axis=1):
        s.set(y)
        y += 1

    print img[:]

    for s in img.iterate(SliceIterator(None, axis=2)):
        print s


    B = Image(N.zeros((3, 4, 5)))
    B.from_slice_iterator(SliceIterator(img))
    print B[:]

    B = Image(N.zeros((4, 3, 5)))
    B.from_slice_iterator(SliceIterator(img), axis=1)
    print B[:]

    B = Image(N.zeros((3, 5, 4)))
    B.from_slice_iterator(SliceIterator(img, axis=2), axis=1)
    print B[:]
    

    B.from_iterator(img.iterate(SliceIterator(None, axis=2)),
                    SliceIterator(None, axis=1))
    print B[:]
    

    x = 0
    for s in B.slice_iterator(mode='w'):
        s.set(x)
        x += 1

    print B[:]
    print "========"

    parcelmap = N.asarray([[0,0,0,1,2],[0,0,1,1,2],[0,0,0,0,2]])
    parcelseq = ((1, 2), (0,), (2,))
    i = SliceParcelIterator(B, parcelmap, parcelseq)
    for n in i:
        print n


    y = 0
    for s in B.slice_iterator(axis=1, mode='w'):
        s.set(y)
        y += 1

    parcelmap = N.asarray([[0,0,0,1,2],[0,0,1,1,2],[0,0,0,0,2]])
    parcelseq = ((1, 2), (0,) , (2,))
    i = SliceParcelIterator(B, parcelmap, parcelseq)
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
