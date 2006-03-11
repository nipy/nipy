import enthought.traits as traits
import numpy as N

import warp
from axis import space, RegularAxis, VoxelAxis, Axis
from coordinate_system import VoxelCoordinateSystem, DiagonalCoordinateSystem, CoordinateSystem
import uuid
from grid_iterators import SliceIterator, ParcelIterator

class SamplingGrid(traits.HasTraits):

    warp = traits.Any()
    shape = traits.ListInt()
    labels = traits.Any()
    labelset = traits.Any()
    itertype = traits.Trait('slice', 'parcel')
    tag = traits.Trait(uuid.Uuid())

    def __init__(self, **keywords):
        traits.HasTraits.__init__(self, **keywords)

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
        elif self.itertype is 'parcel':
            self.iterator = iter(ParcelIterator(self.shape, self.labels, self.labelset))
        return self

    def next(self):
        return self.iterator.next()

class ConcatenatedGrids(SamplingGrid):
    """
    Return a grid formed by concatenating a sequence of grids. Checks are done
    to ensure that the coordinate systems are consistent, as is the shape.

    It returns a grid with the proper shape but no inverse.

    This is most likely the kind of grid to be used for fMRI images.
    """
    grids = traits.List()
    concataxis = traits.Str('concat')

    def __init__(self, grids, **keywords):

        traits.HasTraits.__init__(self, grids=grids, **keywords)
        SamplingGrid.__init__(self, shape=self.shape, warp=self.warp)

    def _grids_changed(self):
        n = len(self.grids)
        self.shape = [n] + self.grids[0].shape

        # check shapes are identical
    
        s = self.grids[0].shape
        check = N.sum([self.grids[i].shape == s for i in range(n)])
        if not check:
            raise ValueError, 'shape must be the same in ConcatenatedGrids'

        # check input coordinate systems are identical
    
        ic = self.grids[0].warp.input_coords
        check = N.sum([self.grids[i].warp.input_coords == ic for i in range(n)])
        if not check:
            raise ValueError, 'input coordinate systems must be the same in ConcatenatedGrids'

        # check output coordinate systems are identical
    
        oc = self.grids[0].warp.output_coords
        check = N.sum([self.grids[i].warp.output_coords == oc for i in range(n)])
        if not check:
            raise ValueError, 'output coordinate systems must be the same in concatenate_grids'

        def _warp(x):
            try:
                I = x[0].view(N.Int)
                X = x[1:]
                v = N.zeros(x.shape[1:], N.Float)
                for j in I.shape[0]:
                    v[j] = self.grids[I[j]].warp.map(X[j])
                return v

            except:
                i = int(x[0])
                x = x[1:]
                return self.grids[i].warp(x)
            
        newaxis = Axis(name=self.concataxis)
        newin = CoordinateSystem('%s:%s' % (ic.name, self.concataxis),
                                 [newaxis] + ic.axes)
        newout = CoordinateSystem('%s:%s' % (oc.name, self.concataxis),
                                 [newaxis] + oc.axes)

        self.warp = warp.Warp(newin, newout, _warp)

    def subgrid(self, i):
        return self.grids[i]
                           
class DuplicatedGrids(ConcatenatedGrids):

    step = traits.Float(1.)
    start = traits.Float(0.)

    def __init__(self, grid, j, **keywords):
        ConcatenatedGrids.__init__(self, [grid]*j, **keywords)

    def _grids_changed(self):
        ConcatenatedGrids._grids_changed(self)
        ndim = len(self.shape)
        t = N.zeros((ndim + 1,)*2, N.Float)
        t[0:(ndim-1),0:(ndim-1)] = self.grids[0].warp.transform[0:(ndim-1),0:(ndim-1)]
        t[0:(ndim-1),ndim] = self.grids[0].warp.transform[0:(ndim-1),(ndim-1)]
        t[(ndim-1),(ndim-1)] = self.step
        t[(ndim-1),ndim] = self.start
        t[ndim,ndim] = 1.
        w = warp.Affine(self.warp.input_coords, self.warp.output_coords, t)

def fromStartStepLength(names=space, shape=[], start=[], step=[]):    
    """
    Generate a SampingGrid instance from sequences of names, shape, start and step.
    """
    indim = []
    outdim = []

    ndim = len(names)

    # fill in default step size
    step = N.array(step)
    step = N.where(step, step, 1.)

    outdim = [RegularAxis(name=names[i], length=shape[i], start=start[i], step=step[i]) for i in range(ndim)]
    indim = [VoxelAxis(name=names[i], length=shape[i]) for i in range(ndim)]
    
    input_coords = VoxelCoordinateSystem('voxel', indim)
    output_coords = DiagonalCoordinateSystem('world', outdim)
    transform = output_coords.transform()
    _warp = warp.Affine(input_coords, output_coords, transform)
    return SamplingGrid(warp=_warp, shape=list(shape))

def IdentityGrid(shape=(), names=space):
    """
    Return the identity SamplingGrid based on a given shape.
    """
    
    ndim = len(shape)
    w = warp.IdentityWarp(ndim, names=names)
    return SamplingGrid(shape=list(shape), warp=w)
    if len(names) != ndim:
        raise ValueError, 'shape and number of axisnames do not agree'

    return fromStartStepLength(names=names, shape=shape, start=[0]*ndim, step=[1]*ndim)

def matlab2python(grid):
    shape = grid.shape[::-1]
    _warp = warp.matlab2python(grid.warp)
    return SamplingGrid(shape=shape, warp=_warp)

def python2matlab(grid):
    shape = grid.shape[::-1]
    _warp = warp.python2matlab(grid.warp)
    return SamplingGrid(shape=shape, warp=_warp)

