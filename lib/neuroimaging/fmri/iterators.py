from neuroimaging.core.reference.iterators import SliceIterator, SliceParcelIterator

class fMRISliceIterator(SliceIterator):
    """
    Instead of iterating over slices of a 4d file -- return slices of
    timeseries.
    """

    def __init__(self, end):
        SliceIterator.__init__(self, end, axis = 1)
        self.nframe = self.end[0]


class fMRISliceParcelIterator(SliceParcelIterator):
    "Return parcels of timeseries within slices."

    def __init__(self, parcelmap, parcelseq, shape):
        nframe = shape[0]
        SliceParcelIterator.__init__(self, parcelmap, parcelseq)
        self.nframe = nframe

    def next(self):
        value = SliceParcelIterator.next(self)
        return SliceParcelIterator.Item(
            value.label, value.where,(slice(0,self.nframe), value.slice))

