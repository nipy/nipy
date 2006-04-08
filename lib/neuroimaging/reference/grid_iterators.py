# Standard imports
import sets

# External imports
import numpy as N
import enthought.traits as traits

# Package imports
from slicer import Slicer

class IteratorNext(traits.HasTraits):
    type = traits.Trait('slice', 'parcel', 'slice/parcel', 'all')

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
        self.allslice = [slice(self.start[i],
                               self.end[i],
                               self.step[i]) for i in range(self.nslicedim)]

    def next(self):
        _slice, isend = Slicer.next(self)
        return SliceIteratorNext(slice=_slice, type='slice')

class AllSliceIterator(Slicer):

    type = traits.Str('all')
    parallel = traits.false

    def __iter__(self):
        self.isend = False
        return self

    def _parallel_changed(self):
        if self.parallel:
            a, b = prange(self.shape[0])

    def __init__(self, shape, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.shape = shape
        self.isend = False

    def next(self):
        if self.isend:
            raise StopIteration
        _slice = slice(0, self.shape[0], 1)
        self.isend = True
        return SliceIteratorNext(slice=_slice, type='slice')

class ParcelIteratorNext(IteratorNext):
    type = traits.Str('parcel')
    where = traits.Any()
    label = traits.Any()

class ParcelIterator:
    
    labels = traits.Any()
    labelset = traits.Any()
    multilabels = traits.false # are the labels one key or are they a sequence?
    
    def __init__(self, labels, keys, **keywords):
        self.labels = labels
        if not isinstance(self.labels, N.ndarray):
            self.labels = N.array(self.labels)

        self.labelset = list(sets.Set(keys))

        try:
            self.nlabel = len(self.labelset[0])
            self.multilabels = True
        except:
            self.multilabels = False
            pass

        if not self.multilabels:
            self.labels.shape = N.product(self.labels.shape)
        else:
            self.labels.shape = (self.labels.shape[0],
                                 N.product(self.labels.shape[1:]))



    def __iter__(self):
        self.labelset = iter(self.labelset)
        return self

    def next(self):

        label = self.labelset.next()
        if not self.multilabels:
            wherelabel = N.equal(self.labels, label)
        else:
            wherelabel = 1
            for i in range(self.nlabel):
                wherelabel = wherelabel * N.equal(self.labels[i], label[i])
            
        return ParcelIteratorNext(label=label,
                                  where=wherelabel)

        
class SliceParcelIteratorNext(IteratorNext):
    type = traits.Str('slice/parcel')
    where = traits.Any()
    label = traits.Any()
    slice = traits.Any()

class SliceParcelIterator:
    
    """
    This iterator assumes that labels is a list of lists (or an array)
    and the keys is a sequence of length labels.shape[0] (=len(labels)).
    It then goes through the each element in the sequence
    of labels returning where the unique elements are from keys.
    """

    labelset = traits.Any()
    labels = traits.Any()
    
    def __init__(self, labels, keys, **keywords):
        self.labels = labels
        self.labelset = list(keys)

        if len(self.labels) != len(self.labelset):
            raise ValueError, 'labels and labelset do not have the same length'

    def __iter__(self):
        self.curslice = -1
        self.labelset = iter(self.labelset)
        return self

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
            
        return SliceParcelIteratorNext(label=label,
                                       where=wherelabel,
                                       slice=self.curslice)

        
