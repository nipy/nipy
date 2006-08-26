"""
This module provides a set of classes to be used to iterate over sampling grids.
"""

import operator

# External imports
import numpy as N
from protocols import haslength

itertypes = ("slice", "parcel", "slice/parcel")

class SliceIterator (object):
    """
    This class is an iterator that steps through the slices of an
    N-dimensional array of shape shape, along a particular axis, with a given
    step and an optional start.

    The attribute nslicedim determines how long a slice is returned, only
    step[0:nslicedim] and start[0:nslicedim] is used, where self.step
    defaults to [1]*(nslicedim).

    More than one slice can be output at a time, using nslice.
    """
    class Item (object):
        "iterator item"
        def __init__(self, slice): 
            self.slice = slice
            self.type = "slice"

    def __init__(self, end, axis=0, start=None, step=None, nslicedim=0,
      nslice=1):
        self.end = tuple(end)
        self.axis = axis
        self.nslice = nslice
        self.nslicedim = max(nslicedim, self.axis+1)        
        if start is None:
            self.start = N.zeros(self.nslicedim, N.int32)
        else:
            self.start = N.asarray(start)
        if step is None:
            self.step = N.ones(self.nslicedim, N.int32)
        else:
            self.step = N.asarray(step)

        self.end = N.asarray(end)
        self.ndim = len(self.end)
        self.allslice = [slice(self.start[i], self.end[i], self.step[i]) \
                         for i in range(self.nslicedim)]

        self.slice = self.start[self.axis]
        self.last = self.end[self.axis]
        self.type = "slice"
        self._isend = False

    def next(self):
        if self._isend:
            raise StopIteration
        _slices = []
        for i in range(self.nslicedim):
            if self.axis != i:
                _slice = slice(self.start[i], self.end[i], self.step[i])
                _slices.append(_slice)
            else:
                _slice = slice(self.slice,
                               self.slice + self.nslice * self.step[i],
                               self.step[i])
                self.slice += self.nslice * self.step[i]
                _slices.append(_slice)

        if self.slice >= self.last: self._isend = True
        return SliceIterator.Item(_slices)



class ParcelIterator (object):
    """
    Iterates over subsets of an image grid.  Each iteration returns a boolean
    mask with the same shape as the grid indicating the elements of the current
    subset.
    
    >>> from numpy import *
    >>> parcelmap = asarray([[0,0,0,1,2],[0,0,1,1,2],[0,0,0,0,2]])
    >>> parcelseq = ((1,2),0)
    >>> from neuroimaging.core.reference.iterators import ParcelIterator
    >>> i = ParcelIterator(parcelmap,parcelseq)
    >>> for n in i: print n
    ...
    Item(label=(1, 2), where=array([[False, False, False, True, True],
           [False, False, True, True, True],
           [False, False, False, False, True]], dtype=bool))
    Item(label=(0,), where=array([[True, True, True, False, False],
           [True, True, False, False, False],
           [True, True, True, True, False]], dtype=bool))
    >>>
    """

    class Item (object):
        "iterator item"
        def __init__(self, label, where):
            self.label, self.where = tuple(label), where
            self.type = "parcel"

        def __repr__(self):
            return "%s(label=%s, where=%s)"%\
             (self.__class__.__name__, `self.label`,`self.where`)


    def __init__(self, parcelmap, parcelseq=None):
        self.parcelmap = N.asarray(parcelmap)
        "numpy.ndarray of ints defining region(s) for different parcels"
        if parcelseq is not None: 
            self.parcelseq = tuple(parcelseq)
        else:
            self.parcelseq = N.unique(self.parcelmap.flat)
        """
        Sequence of ints and/or sequences of ints indicating which parcels
        and/or collections parcels to iterate over.
        """


        iter(self)
        self._labeliter = iter(self.parcelseq)

    def __iter__(self):
        return self

    def next(self):
        label = self._labeliter.next()
        if not haslength(label): label = (label,)
        wherelabel = reduce(operator.or_,
          [N.equal(self.parcelmap, lbl) for lbl in label])
        return ParcelIterator.Item(label, wherelabel)


       

class SliceParcelIterator (object):
    """
    SliceParcelIterator iterates over a different (or potentially identical)
    collection of subsets (parcels) for each slice of the parcelmap.
    parcelseq is a sequence of ints and/or sequences of ints indicating which
    subset of each slice of parcelmap to return.  Each iteration returns a
    boolean mask with the same shape as a slice of parcelmap indicating the
    elements of that slice's subset.

   >>> from numpy import *
   >>> parcelmap = asarray([[0,0,0,1,2],[0,0,1,1,2],[0,0,0,0,2]])
   >>> parcelseq = ((1,2),0,2)
   >>> from neuroimaging.core.reference.iterators import SliceParcelIterator
   >>> i = SliceParcelIterator(parcelmap,parcelseq)
   >>> for n in i: print n
   ...
   Item(label=(1, 2), where=array([False, False, False, True, True], dtype=bool))
   Item(label=(0,), where=array([True, True, False, False, False], dtype=bool))
   Item(label=(2,), where=array([False, False, False, False, True], dtype=bool))
   >>>

    """
     
    class Item (ParcelIterator.Item):
        "iterator item"
        def __init__(self, label, where, _slice):
            ParcelIterator.Item.__init__(self, label, where)
            self.slice = _slice
            self.type = "slice/parcel"


    def __init__(self, parcelmap, parcelseq):
        self.parcelmap = parcelmap
        if len(parcelmap) != len(parcelseq):
            raise ValueError, 'parcelmap and parcelseq must have the same length'
        self.parcelseq = parcelseq
        self._loopvars = iter(enumerate(zip(self.parcelmap, self.parcelseq)))


    def __iter__(self):
        return self

    def next(self):
        index, (mapslice,label) = self._loopvars.next()
        item = ParcelIterator(mapslice, (label,)).next()
        return SliceParcelIterator.Item(item.label,item.where,index)

        # get rid of index and type from SliceParcelIterator.Item, then do this:
        #return ParcelIterator.Item(mapslice, (label,)).next()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
