from attributes import readonly

from neuroimaging.reference.iterators import SliceIterator, \
  ParcelIterator, SliceParcelIterator, SliceParcelIteratorNext

##############################################################################
class fMRISliceIterator(SliceIterator):
    """
    Instead of iterating over slices of a 4d file -- return slices of
    timeseries.
    """
    class nframe (readonly): get=lambda _,s: s.end[0]
    def __init__(self, end, **kwargs):
        kwargs["axis"]=1
        SliceIterator.__init__(self, end, **kwargs)


##############################################################################
class fMRISliceParcelIterator(SliceParcelIterator):
    "Return parcels of timeseries within slices."
    class nframe (readonly): implements=int

    def __init__(self, labels, labelset, nframe, **keywords):
        SliceParcelIterator.__init__(self, labels, labelset, **keywords)
        self.nframe = nframe

    def next(self):
        value = SliceParcelIterator.next(self)
        value.slice = [slice(0,self.nframe,1), value.slice]
        return value
