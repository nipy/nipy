"""
Template ROI class for BrainSTAT
"""

import numpy as N
import gc, sets, UserList

class ROI:
    """
    This is the basic ROI class, which we model as basically
    a function defined on Euclidean space, i.e. R^3. For practical
    purposes, this function is evaluated on the range of a Warp
    instance.
    """

    def __init__(self, coordinate_system):
        self.coordinate_system = coordinate_system

class ContinuousROI(ROI):
    """
    Create an ROI with a binary function in a given coordinate system.
    """
    ndim = 3
    def __init__(self, coordinate_system, bfn, args={}, ndim=ndim):
        ROI.__init__(self, coordinate_system)

        self.args = args
        self.bfn = bfn
        if not callable(bfn):
            raise ValueError, 'first argument to ROI should be a callable function.'
        # test whether it executes properly

        try:
            x = bfn(N.array([[0.,] * self.ndim]))
        except:
            raise ValueError, 'binary function bfn in ROI failed on ' + `[0.] * self.ndim`

    def __call__(self, real):
        return not_equal(self.bfn(real, **self.args), 0)

    def todiscrete(self, voxels):
        """
        Return a DiscreteROI instance at the voxels in the ROI.
        """
        v = []
        for voxel in voxels:
            if self(voxel):
                v.append(voxel)
        return DiscreteROI(coordinate_system, v)
    
    def togrid(self, grid):
        """
        Return a SamplingGridROI instance at the voxels in the ROI.
        """
        v = []
        for voxel in iter(grid):
            if self(voxel):
                v.append(voxel)
        return SamplingGridROI(coordinate_system, v, grid)

class DiscreteROI(ROI):

    def __init__(self, coordinate_system, voxels):
        ROI.__init__(self, coordinate_system)
        self.voxels = sets.Set(voxels)

    def __iter__(self):
        self.voxels = iter(self.voxels)
        return self

    def next(self):
        return self.voxels.next()

    def pool(self, fn, **extra):
        '''
        Pool data from an image over the ROI -- return fn evaluated at each voxel.
        '''
        v = []

        for voxel in iterator:
            v.append(fn(voxel))
        return v
        
    def feature(self, fn, **extra):
        """
        Return a feature of an image within the ROI. Feature args are 'args', while **extra are for the readall method. Default is to reduce a ufunc over the ROI. Any other operations should be able to ignore superfluous keywords arguments, i.e. use **extra.
        """

        pooled_data = self.pool(fn)
        return N.mean(pooled_data)
        
    def __add__(self, other):
        if isinstance(other, DiscreteROI):
            if other.coordinate_system == self.coordinate_system:
                voxels = sets.Set(self.voxels) + sets.Set(other.voxels)
                return DiscreteROI(self.coordinate_system, voxels)
            else:
                raise ValueError, 'coordinate systems do not agree in union of DiscreteROI'
        else:
            raise NotImplementedError, 'only unions of DiscreteROIs with themselves are implemented'

class SamplingGridROI(DiscreteROI):

    def __init__(self, coordinate_system, voxels, grid):
        ROI.__init__(self, coordinate_system)
        self.grid = grid
        self.voxels = sets.Set(voxels)
        # we assume that voxels are (i,j,k) indices?

    def pool(self, image, **extra):
        '''
        Pool data from an image over the ROI -- return fn evaluated at each voxel.
        '''
        v = []
        if image.grid != self.grid:
            raise ValueError, 'to pool an image over a SamplingGridROI the grids must agree'

        tmp = image.readall()
        for voxel in iterator:
            v.append(tmp[voxel])

        del(tmp); gc.collect()
        return v
        
    def __mul__(self, other):
        if isinstance(other, SamplingGridROI):
            if other.grid == self.grid:
                voxels = self.voxels.intersect(other.voxels)
                return SamplingGridROI(self.coordinate_system, voxels, grid)
            else:
                raise ValueError, 'grids do not agree in union of SamplingGridROI'
        else:
            raise NotImplementedError, 'only unions of SamplingGridROIs with themselves are implemented'

    def mask(self):
        m = N.zeros(self.grid.shape, N.Int)
        for v in self.voxels:
            m[v] = 1.
        return m
    
class ROIall(SamplingGridROI):
    """
    An ROI for an entire grid. Save time by avoiding compressing, etc.
    """

    def mask(self, warp, **keywords):
        try:
            warp = image.spatial_warp
        except:
            warp = image # is it a warp?
        return N.ones(warp.shape)

    def pool(self, image):
        tmp = image.readall()
        tmp.shape = N.product(tmp)

def ROIspherefn(center, radius):
    def test(real):
        diff = array([real[i] - center[i] for i in range(real.shape[0])])
        return (sum(diff**2) < radius**2)
    return test

def ROIellipsefn(center, form, a = 1.0):
    """
    Ellipse determined by regions where a quadratic form is <= a. The
    quadratic form is given by the inverse of the 'form' argument, so
    a sphere of radius 10 can be specified as {'form':10**2 * identity(3), 'a':1} or {'form':identity(3), 'a':100}.

    Form must be positive definite.
    """
    from BrainSTAT.Modules.KernelSmooth import normsq
    from numpy.linalg import cholesky_decomposition, inverse
    _cholinv = cholesky_decomposition(inverse(form))
    ndim = array(center).shape[0]

    def test(real):
        _real = 1. * real
        for i in range(ndim):
            _real[i] = _real[i] - center[i]
        _shape = _real.shape
        _real.shape = _shape[0], product(_shape[1:])

        X = dot(_cholinv, _real)
        d = sum(X**2)
        d.shape = _shape[1:]
        value = less_equal(d, a)

        del(_real); del(X); del(d)
        gc.collect()
        return value
    return test

def ROIfromArraySamplingGrid(data, grid):
    """
    Return a SamplingGridROI from an array (data) on a grid.
    interpolation. Obvious ways to extend this.
    """

    if grid.shape != data.shape:
        raise ValueError, 'grid shape does not agree with data shape'
    voxels = nonzero(data)
    coordinate_system = image.grid.output_coordinate_system
    return SamplingGridROI(coordinate_system, voxels, grid)

class ROISequence(UserList.UserList):
    pass
