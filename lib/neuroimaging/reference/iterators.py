import operator

# External imports
import numpy as N
from attributes import attribute, readonly, constant, clone
from protocols import Sequence, haslength

itertypes = ("slice", "parcel", "slice/parcel")


##############################################################################


##############################################################################
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
    class type (constant): default="slice"
    class end (readonly):
        def set(_, self, value): readonly.set(_, self, N.asarray(value))
    class start (readonly):
        def init(_, self): return N.array((0,)*self.nslicedim)
        def set(_, self, value): readonly.set(_, self, N.asarray(value))
    class step (readonly):
        def init(_, self): return N.array((1,)*self.nslicedim)
        def set(_, self, value): readonly.set(_, self, N.asarray(value))
    class axis (readonly): default=0
    class nslicedim (readonly): implements=int
    class nslice (readonly): default=1
    class ndim (readonly): get=lambda _,s: len(s.end)
    class _isend (attribute): default=False

    #-------------------------------------------------------------------------
    class Item (object):
        "iterator item"
        class type (constant): default="slice"
        class slice (attribute): implements=Sequence,slice
        def __init__(self, slice): self.slice = slice

    #-------------------------------------------------------------------------
    def __init__(self, end, start=None, step=None, axis=None, nslicedim=None,
      nslice=None):
        self.end = tuple(end)
        for attr in ("start","step","axis","nslice"):
            if locals()[attr] is not None: setattr(self, attr, locals()[attr])

        self.nslicedim = max(nslicedim, self.axis+1)

        self.allslice = [slice(self.start[i], self.end[i], self.step[i]) \
                         for i in range(self.nslicedim)]

        self.slice = self.start[self.axis]
        self.last = self.end[self.axis]

    #-------------------------------------------------------------------------
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


##############################################################################
class ParcelIterator (object):
    """
    Iterates over subsets of an image grid.  Each iteration returns a boolean
    mask with the same shape as the grid indicating the elements of the current
    subset.
    
    >>> from numpy import *
    >>> parcelmap = asarray([[0,0,0,1,2],[0,0,1,1,2],[0,0,0,0,2]])
    >>> parcelseq = ((1,2),0)
    >>> i = ParcelIterator(parcelmap,parcelseq) 
    >>> for n in i: print n
    ...
    Item(label=(1, 2), where=array([[False, False, False, True, True],
           [False, False, True, True, True],
           [False, False, False, False, True]], dtype=bool))
    Item(label=(0,), where=array([[True, True, True, False, False],
           [True, True, False, False, False],
           [True, True, True, True, False]], dtype=bool))
    """

    class parcelmap (readonly):
        "numpy.ndarray of ints defining region(s) for different parcels"
        default=N.asarray(())
        def set(_, self, value):
            readonly.set(_, self, N.asarray(value))

    class parcelseq (readonly):
        """
        Sequence of ints and/or sequences of ints indicating which parcels
        and/or collections parcels to iterate over.
        """
        implements=Sequence
        def init(att, self):
            return N.unique(self.parcelmap.flat)

    #-------------------------------------------------------------------------
    class Item (object):
        "iterator item"

        class type (constant): default="parcel"
        class label (readonly): implements=tuple,int
        class where (readonly): pass

        def __init__(self, label, where):
            self.label, self.where = tuple(label), where

        def __repr__(self):
            return "%s(label=%s, where=%s)"%\
             (self.__class__.__name__, `self.label`,`self.where`)

    #-------------------------------------------------------------------------
    def __init__(self, parcelmap, parcelseq=None):
        self.parcelmap = parcelmap
        if parcelseq is not None: self.parcelseq = tuple(parcelseq)
        self._labeliter = iter(self.parcelseq)

    #-------------------------------------------------------------------------
    def next(self):
        label = self._labeliter.next()
        if not haslength(label): label = (label,)
        wherelabel = reduce(operator.or_,
          [N.equal(self.parcelmap, lbl) for lbl in label])
        return ParcelIterator.Item(label, wherelabel)


       
##############################################################################
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
    >>> i = SliceParcelIterator(parcelmap,parcelseq) 
    >>> for n in i: print n
    ...
    Item(label=(1, 2), where=array([False, False, False, True, True], dtype=bool))
    Item(label=(0,), where=array([True, True, False, False, False], dtype=bool))
    Item(label=(2,), where=array([False, False, False, False, True], dtype=bool))
    """

    clone(ParcelIterator.parcelmap)
    clone(ParcelIterator.parcelseq)
     
    #-------------------------------------------------------------------------
    class Item (ParcelIterator.Item):
        "iterator item"

        class type (constant): default="slice/parcel"
        class slice (readonly): "slice index"; implements=tuple,int

        def __init__(self, label, where, _slice):
            ParcelIterator.Item.__init__(self, label, where)
            self.slice = _slice

    #-------------------------------------------------------------------------
    def __init__(self, parcelmap, parcelseq):
        self.parcelmap = parcelmap
        if len(parcelmap) != len(parcelseq):
            raise ValueError, 'parcelmap and parcelseq must have the same length'
        self.parcelseq = parcelseq
        self._loopvars = iter(enumerate(zip(self.parcelmap, self.parcelseq)))

    #-------------------------------------------------------------------------
    def next(self):
        index, (mapslice,label) = self._loopvars.next()
        item = ParcelIterator(mapslice, (label,)).next()
        return SliceParcelIterator.Item(item.label,item.where,index)

        # get rid of index and type from SliceParcelIterator.Item, then do this:
        #return ParcelIterator.Item(mapslice, (label,)).next()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
