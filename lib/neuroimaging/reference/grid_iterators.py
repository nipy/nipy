# Standard imports
import sets

# External imports
import numpy as N
import enthought.traits as traits
from attributes import readonly, constant

# Package imports
from neuroimaging import haslength
from neuroimaging.reference.slicer import Slicer

itertypes = ("slice", "parcel", "slice/parcel", "all")

##############################################################################
class Iterator (object):
    "Iterator protocol"
    def next(self): pass


##############################################################################
class SliceIteratorNext (object):
    class type (constant): default="slice"
    class slice (readonly): pass
    def __init__(self, slice): self.slice = slice


##############################################################################
class SliceIterator(Slicer):

    parallel = traits.false

    #-------------------------------------------------------------------------
    def _parallel_changed(self):
        if self.parallel: a, b = prange(self.shape[0])

    #-------------------------------------------------------------------------
    def __init__(self, shape, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        Slicer.__init__(self, shape, **keywords)
        self.allslice = [slice(self.start[i],
                               self.end[i],
                               self.step[i]) for i in range(self.nslicedim)]

    #-------------------------------------------------------------------------
    def next(self):
        _slice, isend = Slicer.next(self)
        return SliceIteratorNext(_slice)


##############################################################################
class AllSliceIterator(Slicer):

    type = traits.Str('all')
    parallel = traits.false

    #-------------------------------------------------------------------------
    def __iter__(self):
        self.isend = False
        return self

    #-------------------------------------------------------------------------
    def _parallel_changed(self):
        if self.parallel: a, b = prange(self.shape[0])

    #-------------------------------------------------------------------------
    def __init__(self, shape, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.shape = shape
        self.isend = False

    #-------------------------------------------------------------------------
    def next(self):
        if self.isend:
            raise StopIteration
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
class ParcelIterator (traits.HasTraits):
    labels = traits.Any()
    labelset = traits.Any()

    #-------------------------------------------------------------------------
    def __init__(self, labels, keys, **keywords):
        self.labels = N.asarray(labels)
        self.labelset = list(sets.Set(keys))
        self.labels.shape = haslength(self.labelset[0]) and\
          (self.labels.shape[0], N.product(self.labels.shape[1:])) or\
          N.product(self.labels.shape)

    #-------------------------------------------------------------------------
    def __iter__(self):
        self.labelset = iter(self.labelset)
        return self

    #-------------------------------------------------------------------------
    def next(self):
        keys = self.labelset.next()
        if not haslength(keys): keys = (keys,)
        wherelabel = N.product([N.equal(label, key)\
          for label,key in zip(self.labels, keys)])
        return ParcelIteratorNext(keys, wherelabel)

 
##############################################################################
class SliceParcelIteratorNext (ParcelIteratorNext, SliceIteratorNext):
    class type (constant): default="slice/parcel"
    def __init__(self, label, where, slice):
        SliceIteratorNext.__init__(slice)
        ParcelIteratorNext.__init__(label, where)

       
##############################################################################
class SliceParcelIterator (traits.HasTraits):
    """
    This iterator assumes that labels is a list of lists (or an array)
    and the keys is a sequence of length labels.shape[0] (=len(labels)).
    It then goes through the each element in the sequence
    of labels returning where the unique elements are from keys.
    """
    labelset = traits.Any()
    labels = traits.Any()
            
    #-------------------------------------------------------------------------
    def __init__(self, labels, keys, **keywords):
        self.labels = labels
        self.labelset = list(keys)

        if len(self.labels) != len(self.labelset):
            raise ValueError, 'labels and labelset do not have the same length'

    #-------------------------------------------------------------------------
    def __iter__(self):
        self.curslice = -1
        self.labelset = iter(self.labelset)
        return self

    #-------------------------------------------------------------------------
    def next(self):
        try:
            label = self.curlabelset.next()
        except:
            self.curlabelset = iter(self.labelset.next())
            label = self.curlabelset.next()
            self.curslice += 1
            pass

        self.curlabels = self.labels[self.curslice]

        if not isinstance(self.curlabels, N.ndarray):
            self.curlabels = N.array(self.curlabels)
            
        self.curlabels.shape = N.product(self.curlabels.shape)
        wherelabel = N.equal(self.curlabels, label)
        return SliceParcelIteratorNext(label, wherelabel, self.curslice)

        
