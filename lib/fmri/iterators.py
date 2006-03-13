import neuroimaging.reference.grid_iterators as iterators
import neuroimaging.reference.grid as grid
import enthought.traits as traits

class fMRISliceIterator(grid.SliceIterator):
    """
    Instead of iterating over slices of a 4d file -- return slices
    of timeseries.
    """

    nframe = traits.Int()

    def __init__(self, shape, **keywords):
        grid.SliceIterator.__init__(self, shape[1:], **keywords)
        self.nframe = shape[0]

    def next(self):
        value = grid.SliceIterator.next(self)
        _slice = [slice(0,self.nframe,1), value.slice]
        return iterators.SliceIteratorNext(slice=_slice, type='slice')

class fMRIParcelIterator(grid.ParcelIterator):
    """
    Return parcels of timeseries.
    """


class fMRISliceParcelIterator(grid.SliceParcelIterator):
    """
    Return parcels of timeseries within slices.
    """

    nframe = traits.Int()

    def __init__(self, labels, labelset, shape, **keywords):
        grid.SliceParcelIterator.__init__(self, labels, labelset, **keywords)
        self.nframe = shape[0]

    def next(self):
        value = grid.SliceParcelIterator.next(self)
        _slice = [slice(0,self.nframe,1), value.slice]
        return iterators.SliceParcelIteratorNext(slice=_slice, type=value.type, label=value.label, where=value.where)

