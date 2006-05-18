from attributes import readonly

from neuroimaging.reference.iterators import SliceIterator, \
  ParcelIterator, SliceParcelIterator, SliceParcelIteratorNext

##############################################################################
class fMRISliceIterator(SliceIterator):
    """
    Instead of iterating over slices of a 4d file -- return slices
    of timeseries.
    """
    class nframe (readonly): get=lambda _,s: s.end[0]
    def __init__(self, end, **kwargs):
        kwargs["axis"]=1
        SliceIterator.__init__(self, end, **kwargs)


##############################################################################
class fMRISliceParcelIterator(SliceParcelIterator):
    "Return parcels of timeseries within slices."
    class nframe (readonly): implements=int

    def __init__(self, labels, labelset, shape, **keywords):
        SliceParcelIterator.__init__(self, labels, labelset, **keywords)
        self.nframe = shape[0]

    def next(self):
        value = SliceParcelIterator.next(self)
        _slice = [slice(0,self.nframe,1), value.slice]
        return SliceParcelIteratorNext(slice=_slice, type=value.type,
          label=value.label, where=value.where)
