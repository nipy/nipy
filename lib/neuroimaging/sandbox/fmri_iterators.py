from neuroimaging.sandbox.old_iterators import SliceIterator, SliceParcelIterator

class fMRISliceIterator(SliceIterator):
    """
    Instead of iterating over slices of a 4d file -- return slices of
    timeseries.
    """

    def __init__(self, shape):
        SliceIterator.__init__(self, shape, axis = 1)
        self.nframe = self.end[0]


class fMRISliceParcelIterator(SliceParcelIterator):
    """Return parcels of timeseries within slices."""

    def __init__(self, parcelmap, parcelseq, shape):
        SliceParcelIterator.__init__(self, parcelmap, parcelseq)
        self.nframe = shape[0]

    def next(self):
        value = SliceParcelIterator.next(self)
        return SliceParcelIterator.Item(
            value.label, value.where,(slice(0,self.nframe), value.slice))

