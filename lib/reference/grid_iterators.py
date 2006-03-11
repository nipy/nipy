import enthought.traits as traits
import sets
import numpy as N
from slicer import Slicer

class IteratorNext(traits.HasTraits):
    type = traits.Trait('slice', 'parcel')

class SliceIteratorNext(IteratorNext):
    slice = traits.Any()

class SliceIterator(Slicer):

    parallel = traits.false

    def _parallel_changed(self):
        if self.parallel:
            a, b = prange(self.shape[0])

    def __init__(self, shape, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        Slicer.__init__(self, shape, **keywords)

    def next(self):
        _slice, isend = Slicer.next(self)
        return SliceIteratorNext(slice=_slice, type='slice')

class ParcelIteratorNext(IteratorNext):
    type = traits.Str('parcel')
    where = traits.Any()
    label = traits.Any()

class ParcelIterator:
    
    labels = traits.Any()
    def __init__(self, shape, labels, keys, **keywords):
        self.labels = labels
        self.labels.shape = N.product(self.labels.shape)
        self.labelset = sets.Set(keys)

    def __iter__(self):
        self.labelset = iter(self.labelset)
        return self

    def next(self, callnext=False):

        label = self.labelset.next()
        where = N.equal(self.labels, label)
            
        return ParcelIteratorNext(label=label,
                                  where=where)

        
