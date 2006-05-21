from neuroimaging.reference.iterators import SliceIterator, SliceIteratorNext,\
  ParcelIterator, SliceParcelIterator, SliceParcelIteratorNext
import enthought.traits as traits

class fMRISliceIterator(SliceIterator):
    """
    Instead of iterating over slices of a 4d file -- return slices
    of timeseries.
    """
    nframe = traits.Int()

    def __init__(self, shape, **keywords):
        SliceIterator.__init__(self, shape, axis=1, **keywords)
        self.nframe = shape[0]

    def next(self):
        value = iterators.SliceIterator.next(self)
        return iterators.SliceIteratorNext(slice=value.slice, type='slice')


class fMRIParcelIterator(iterators.ParcelIterator):
    """
    Return parcels of timeseries.
    """


class fMRISliceParcelIterator(iterators.SliceParcelIterator):
    """
    Return parcels of timeseries within slices.
    """

    nframe = traits.Int()

    def __init__(self, labels, labelset, shape, **keywords):
        iterators.SliceParcelIterator.__init__(self, labels, labelset, **keywords)
        self.nframe = shape[0]

    def next(self):
        value = iterators.SliceParcelIterator.next(self)
        _slice = [slice(0,self.nframe,1), value.slice]
        return iterators.SliceParcelIteratorNext(slice=_slice, type=value.type, label=value.label, where=value.where)

