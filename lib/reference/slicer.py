import enthought.traits as TR
from numpy import *

class Slicer(TR.HasTraits):
    ''' This class is an iterator that steps through the slices of a N-dimensional image, with slices of size N-nloopdim.'''

    shape = TR.ListInt()
    nloopdim = TR.Int(1)
    shift = TR.Int()
    ndim = TR.Int()
    type = TR.String('slice')

    def _shape_changed(self):
        self.ndim = len(self.shape)

    def __init__(self, shape, **keywords):
        TR.HasTraits.__init__(self, **keywords)
        self.shape = list(shape)

    def __iter__(self):
        self.isend = False
        self.slice = [0]*self.nloopdim
        self.end = tuple(array(self.shape[0:self.nloopdim]) - 1)
        return self

    def next(self):
        if self.isend:
            raise StopIteration
        value = tuple(self.slice)
        if value == self.end:
            self.isend = True

        self.slice[-1] = self.slice[-1] + 1
        for i in range(self.nloopdim-1,0,-1):
            if self.slice[i] == self.shape[i]:
                self.slice[i] = 0
                self.slice[i-1] = self.slice[i-1] + 1
        value = list(value)
        value[0] = value[0] + self.shift

        if len(value) > 1:
            return tuple(value), self.isend
        else:
            if value[0] > 0:
                value = slice(value[0],value[0]+1)
            else:
                value = slice(0,1)
            return value, self.isend


