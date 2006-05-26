# External imports
import numpy as N
from attributes import attribute, readonly, constant
from protocols import Sequence

# Package imports
from neuroimaging import haslength

itertypes = ("slice", "parcel", "slice/parcel", "all")


##############################################################################
class SliceIteratorNext (object):
    class type (constant): default="slice"
    class slice (readonly): pass
    def __init__(self, slice): self.slice = slice


##############################################################################
class SliceIterator (object):
    """
    This class is an iterator that steps through the slices
    of a N-dimensional array of shape shape, along a particular axis, with a
    given step and an optional start.

    The attribute nslicedim determines how long a slice is returned, only
    step[0:nslicedim] and start[0:nslicedim] is used, where self.step
    defaults to [1]*(nslicedim).

    More than one slice can be output at a time, using nslice.
    """
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
    class type (constant): default="slice"

    #-------------------------------------------------------------------------
    def __init__(self, end, start=None, step=None, axis=None, nslicedim=None,
      nslice=None):
        self.end = tuple(end)
        for attr in ("start","step","axis","nslice"):
            if locals()[attr] is not None: setattr(self, attr, locals()[attr])
        self.nslicedim = max(nslicedim, self.axis+1)

        self.allslice = [slice(self.start[i],
                               self.end[i],
                               self.step[i]) for i in range(self.nslicedim)]

    #-------------------------------------------------------------------------
    def __iter__(self):
        self.isend = False
        self.slice = self.start[self.axis]
        self.last = self.end[self.axis]
        return self

    #-------------------------------------------------------------------------
    def next(self):
        if self.isend:
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

        if self.slice >= self.last: self.isend = True
        return SliceIteratorNext(_slices)


##############################################################################
class AllSliceIterator (object):
    class type (constant): default="all"
    class shape (readonly): implements=tuple
    class isend (attribute): default=False

    #-------------------------------------------------------------------------
    def __init__(self, shape): self.shape = shape

    #-------------------------------------------------------------------------
    def __iter__(self):
        self.isend = False
        return self

    #-------------------------------------------------------------------------
    def next(self):
        if self.isend: raise StopIteration
        _slice = slice(0, self.shape[0], 1)
        self.isend = True
        return SliceIteratorNext(_slice)


##############################################################################
class ParcelIteratorNext (object):
    class type (constant): default="parcel"
    class label (readonly): pass
    class where (readonly): pass
    def __init__(self, label, where): self.label, self.where = label, where


##############################################################################
class ParcelIterator (object):
    class parcelmap (readonly): default=N.asarray(())
    class parcelseq (readonly):
        implements=Sequence
        def init(att, self):
            return N.unique(self.parcelmap.flat)

    #-------------------------------------------------------------------------
    def __init__(self, parcelmap, keys=None):
        self.parcelmap = N.asarray(parcelmap)
        if keys is not None: self.parcelseq = list(set(keys))
        self.parcelmap.shape = haslength(self.parcelseq[0]) and\
          (self.parcelmap.shape[0], N.product(self.parcelmap.shape[1:])) or\
          N.product(self.parcelmap.shape)

    #-------------------------------------------------------------------------
    def __iter__(self):
        for label in self.parcelseq:
            if not haslength(label):
                wherelabel = N.equal(self.parcelmap, label)
            else:
                wherelabel = N.product([N.equal(labeled, label)\
                  for labeled,label in zip(self.parcelmap, label)])
            yield ParcelIteratorNext(label, wherelabel)

 
##############################################################################
class SliceParcelIteratorNext (ParcelIteratorNext, SliceIteratorNext):
    class type (constant): default="slice/parcel"
    def __init__(self, label, where, slice):
        SliceIteratorNext.__init__(slice)
        ParcelIteratorNext.__init__(label, where)

       
##############################################################################
class SliceParcelIterator (ParcelIterator):
    """
    This iterator assumes that parcelmap is a list of lists (or an array)
    and the keys is a sequence of length parcelmap.shape[0] (=len(parcelmap)).
    It then goes through the each element in the sequence
    of labels returning where the unique elements are from keys.
    """
    #-------------------------------------------------------------------------
    def __init__(self, parcelmap, keys, **keywords):
        self.parcelmap = parcelmap
        self.parcelseq = iter(keys)

        if len(parcelmap) != len(parcelseq):
            raise ValueError, 'parcelmap and parcelseq do not have the same length'

    #-------------------------------------------------------------------------
    def __iter__(self):
        self.curslice = -1
        return self

    #-------------------------------------------------------------------------
    def next(self):
        try:
            label = self.curparcelseq.next()
        except:
            self.curparcelseq = iter(self.parcelseq.next())
            label = self.curparcelseq.next()
            self.curslice += 1
            pass

        self.curlabels = self.parcelmap[self.curslice]

        if not isinstance(self.curlabels, N.ndarray):
            self.curlabels = N.array(self.curlabels)
            
        self.curlabels.shape = N.product(self.curlabels.shape)
        wherelabel = N.equal(self.curlabels, label)
        return SliceParcelIteratorNext(label, wherelabel, self.curslice)
