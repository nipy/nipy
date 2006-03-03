import warp
import enthought.traits as traits
from slicer import Slicer

class SamplingGrid(traits.HasTraits):

    warp = traits.Any()
    shape = traits.ListInt()
    labels = traits.Any()
    itertype = traits.Trait('slice', 'labelled slice', 'voxel list')

    def __init__(self, **keywords):
        self.args = args
        traits.HasTraits.__init__(self, **keywords)
        self.build = builder

    def range(self):
        """
        Return the coordinate values of a SamplingGrid.

        """
       
        tmp = indices(self.shape)
        _shape = tmp.shape
        tmp.shape = (self.warp.input_coords.ndim, product(self.shape))
        _tmp = self.map(tmp)
        _tmp.shape = _shape
        return _tmp 

    def __iter__(self):
        if self.itertype is 'slice':
            self.iterator = iter(SliceIterator(self.shape))
        if self.itertype is 'labelled slice':
            self.iterator = iter(LabelledSliceIterator(self.shape, self.labels))

    def next(self):
        return self.iterator.next()

class IteratorNext(traits.HasTraits):
    type = traits.Trait('slice', 'labelled slice', 'voxel list')

class SliceIteratorNext(IteratorNext):

    slice = traits.Any()

class SliceIterator(Slicer):

    parallel = traits.false

    def __init__(self, **keywords):
        traits.HasTraits.__init__(**keywords)
        if self.parallel:
            a, b = prange(self.shape[0])
        Slicer.__init__(**keywords)

    def next(self):
        slice, isend = Slicer.next()
        return SliceIteratorNext(slice=slice, type='slice')

class LabelledSliceIteratorNext(SliceIterator):

    labels = traits.Any()

class LabelledSliceIterator(SliceIterator):

    parallel = traits.false

    def __init__(self, labels, **keywords):
        self.labels = labels
        SliceIterator.__init__(**keywords)

    def __iter__(self):
        SliceIterator.__iter__(self)
        iter(self.labels)
        self._outshape = self.shape[-self.nslicedim:]
        self.buffer = zeros(self._outshape, Float).flat
        self._bufshape = self.buffer.shape
        self.newslice = True
        return self

    def next(self):
        SliceIterator.__next__()

        if self.newslice:
            self.labelslice = self.labels.next().flat
            self.newslice = False
            self._labels = iter(list(sets.Set(self.labelslice)))
        else:
            try:
                label = self._labels.next()
                keep = N.equal(self.labelslice, label)
                return LabelledSliceIteratorNext(slice=self.slice, keep=keep,
                                                 type=self.type, newslice=self.newslice)
            except StopIteration:
                self.newslice = True
                return self.next()
    
    
