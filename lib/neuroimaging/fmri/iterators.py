from attributes import readonly

from neuroimaging.reference.iterators import SliceIterator, SliceParcelIterator


class fMRISliceIterator(SliceIterator):
    """
    Instead of iterating over slices of a 4d file -- return slices of
    timeseries.
    """
    class nframe (readonly): "number of frames"; get=lambda _,s: s.end[0]
    def __init__(self, end, **kwargs):
        kwargs["axis"]=1
        SliceIterator.__init__(self, end, **kwargs)



class fMRISliceParcelIterator(SliceParcelIterator):
    "Return parcels of timeseries within slices."

    class nframe (readonly): "number of frames"; implements=int

    def __init__(self, parcelmap, parcelseq, nframe):
        SliceParcelIterator.__init__(self, parcelmap, parcelseq)
        self.nframe = nframe

    def next(self):
        value = SliceParcelIterator.next(self)
        return SliceParcelIterator.Item(
            value.label, value.where,(slice(0,self.nframe), value.slice))

