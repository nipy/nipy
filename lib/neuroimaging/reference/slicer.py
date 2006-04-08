import enthought.traits as traits
import numpy as N

class Slicer(traits.HasTraits):
    '''
    This class is an iterator that steps through the slices
    of a N-dimensional array of shape shape, along a particular axis, with a
    given step and an optional start.

    The attribute nslicedim determines how long a slice is returned, only
    step[0:nslicedim] and start[0:nslicedim] is used, where self.step
    defaults to [1]*(nslicedim).

    More than one slice can be output at a time, using nslice.
    '''

    axis = traits.Int(0)
    end = traits.List()
    step = traits.Any()
    start = traits.Any()
    ndim = traits.Int()
    nslicedim = traits.Int()
    nslice = traits.Int(1)
    
    type = traits.String('slice')

    def _end_changed(self):
        self.ndim = len(self.end)

    def __init__(self, end, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.end = list(end)

        if self.nslicedim < self.axis + 1:
            self.nslicedim = self.axis + 1

        if self.step is None:
            self.step = N.array([1]*self.nslicedim)

        if self.start is None:
            self.start = N.array([0]*self.nslicedim)

    def __iter__(self):
        self.isend = False
        self.slice = self.start[self.axis]
        self.last = self.end[self.axis]
        return self

    def next(self):
        if self.isend:
            raise StopIteration
        _slices = []
        for i in range(self.nslicedim):
            if self.axis != i:
                _slice = slice(self.start[i], self.end[i], self.step[i])
                _slices.append(_slice)
            else:
                _slice = slice(self.slice,
                               self.slice + self.nslice * self.step[i],
                               self.step[i])
                self.slice += self.nslice * self.step[i]
                _slices.append(_slice)

        if self.slice >= self.last:
            self.isend = True

        return _slices, self.isend


