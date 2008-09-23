"""
Template region of interest (ROI) module
"""

__docformat__ = 'restructuredtext'

# FIXME: This module needs some attention. There are no unit tests for it
# so it's hard to say whether it works correctly or not.

import gc

import numpy as np

class ROI:
    """
    This is the basic ROI class, which we model as basically
    a function defined on Euclidean space, i.e. R^3. For practical
    purposes, this function is evaluated on the range of a Mapping
    instance.
    """

    def __init__(self, coordinate_system):
        """
        :Parameters:
            coordinate_system : TODO
                TODO
        """
        self.coordinate_system = coordinate_system

class ContinuousROI(ROI):
    """
    Create an `ROI` with a binary function in a given coordinate system.
    """
    ndim = 3
    def __init__(self, coordinate_system, bfn, args=None, ndim=ndim):
        """
        :Parameters:
            coordinate_system : TODO
                TODO
            bfn : TODO
                TODO
            args : TODO
                TODO
            ndim : ``int``
                TODO
        """
        ROI.__init__(self, coordinate_system)

        if args is None:
            self.args = {}
        else:
            self.args = args
        self.bfn = bfn
        if not callable(bfn):
            raise ValueError(
              'first argument to ROI should be a callable function.')
        # test whether it executes properly

        try:
            bfn(np.array([[0.,] * ndim]))
        except:
            raise ValueError(
              'binary function bfn in ROI failed on ' + `[0.] * ndim`)

    def __call__(self, real):
        """
        :Parameters:
            real : TODO
                TODO

        :Returns: TODO
        """
        return np.not_equal(self.bfn(real, **self.args), 0)

    def todiscrete(self, voxels):        
        """
        Return a `DiscreteROI` instance at the voxels in the ROI.

        :Parameters:
            voxels : TODO
                TODO

        :Returns: `DiscreteROI`
        """
        v = []
        for voxel in voxels:
            if self(voxel):
                v.append(voxel)
        return DiscreteROI(self.coordinate_system, v)
    
    def tocoordmap(self, coordmap):        
        """
        Return a `CoordinateMapROI` instance at the voxels in the ROI.

        :Parameters:
            coordmap : TODO
                TODO

        :Returns: `CoordinateMapROI`
        """
        v = []
        for voxel in iter(coordmap):
            if self(voxel):
                v.append(voxel)
        return CoordinateMapROI(self.coordinate_system, v, coordmap)

class DiscreteROI(ROI):
    """
    TODO
    """
    

    def __init__(self, coordinate_system, voxels):
        """
        :Parameters:
            coordinate_system : TODO
                TODO
            voxels : TODO
                TODO

        """
        ROI.__init__(self, coordinate_system)
        self.voxels = set(voxels)

    def __iter__(self):
        """
        :Returns: ``self``
        """
        self.voxels = iter(self.voxels)
        return self

    def next(self):
        """
        :Returns: TODO
        """
        return self.voxels.next()

    def pool(self, fn, **extra):
        """
        Pool data from an image over the ROI -- return fn evaluated at
        each voxel.

        :Parameters:
            fn : TODO
                TODO
            extras : ``dict``
                TODO

        :Returns: TODO
        """
        return [fn(voxel, **extra) for voxel in self.voxels]
        
    def feature(self, fn, **extra):
        """
        Return a feature of an image within the ROI. Feature args are 'args',
        while ``extra`` are for the readall method. Default is to reduce a ufunc
        over the ROI. Any other operations should be able to ignore superfluous
        keywords arguments, i.e. use ``extra``.

        :Parameters:
            fn : TODO
                TODO
            extra : ``dict
                TODO

        :Returns: `DiscreteROI`

        :Raises ValueError: TODO
        :Raises NotImplementedError: TODO        
        """
        pooled_data = self.pool(fn, **extra)
        return np.mean(pooled_data)
        
    def __add__(self, other):
        if isinstance(other, DiscreteROI):
            if other.coordinate_system == self.coordinate_system:
                voxels = set(self.voxels) + set(other.voxels)
                return DiscreteROI(self.coordinate_system, voxels)
            else:
                raise ValueError(
                  'coordinate systems do not agree in union of DiscreteROI')
        else:
            raise NotImplementedError(
              'only unions of DiscreteROIs with themselves are implemented')

class CoordinateMapROI(DiscreteROI):

    def __init__(self, coordinate_system, voxels, coordmap):
        """
        :Parameters:
            coordinate_system : TODO
                TODO
            voxels : TODO
                TODO
            coordmap : TODO
                TODO
        
        """
        DiscreteROI.__init__(self, coordinate_system, voxels)
        self.coordmap = coordmap
        # we assume that voxels are (i,j,k) indices?

    def pool(self, image):
        """
        Pool data from an image over the ROI -- return fn evaluated at
        each voxel.

        :Parameters:
            image : `image.Image`
                TODO

        :Returns: TODO

        :Raises ValueError: TODO
        """
        if image.coordmap != self.coordmap:
            raise ValueError(
              'to pool an image over a CoordinateMapROI the coordmaps must agree')

        tmp = image.readall()
        v = [tmp[voxel] for voxel in self.voxels]

        del(tmp); gc.collect()
        return v
        
    def __mul__(self, other):
        """
        :Parameters:
            other : TODO
                TODO
        :Returns: `CoordinateMapROI`

        :Raises ValueError: TODO
        :Raises NotImplementedError: TODO
        """
        if isinstance(other, CoordinateMapROI):
            if other.coordmap == self.coordmap:
                voxels = self.voxels.intersect(other.voxels)
                return CoordinateMapROI(self.coordinate_system, voxels, self.coordmap)
            else:
                raise ValueError(
                  'coordmaps do not agree in union of CoordinateMapROI')
        else:
            raise NotImplementedError(
              'only unions of CoordinateMapROIs with themselves are implemented')

    def mask(self):
        """
        :Returns: ``numpy.ndarray`
        """
        m = np.zeros(self.coordmap.shape, np.int32)
        for v in self.voxels:
            m[v] = 1.
        return m
    
class ROIall(CoordinateMapROI):
    """
    An ROI for an entire coordmap. Save time by avoiding compressing, etc.
    """

    def mask(self, image):
        """
        :Parameters:
            image : TODO
                TODO

        :Return: ``numpy.ndarray`
        """
        try:
            mapping = image.spatial_mapping
        except:
            mapping = image # is it a mapping?
        return np.ones(mapping.shape)

    def pool(self, image):
        """
        :Parameters:
            image : `image.Image`

        :Returns: ``None``
        """
        tmp = image.readall()
        tmp.shape = np.product(tmp)

def roi_sphere_fn(center, radius):
    """
    :Parameters:
        center : TODO
            TODO
        radius : TODO
            TODO

    :Returns: TODO
    """
    def test(real):
        diff = np.array([real[i] - center[i] for i in range(real.shape[0])])
        return (sum(diff**2) < radius**2)
    return test

def roi_ellipse_fn(center, form, a = 1.0):
    """
    Ellipse determined by regions where a quadratic form is <= a. The
    quadratic form is given by the inverse of the 'form' argument, so
    a sphere of radius 10 can be specified as
    {'form':10**2 * identity(3), 'a':1} or {'form':identity(3), 'a':100}.

    Form must be positive definite.

    :Parameters:
        form : TODO
            TODO
        a : ``float``
            TODO
            
    :Returns: TODO
    """
    from numpy.linalg import cholesky, inv
    _cholinv = cholesky(inv(form))
    ndim = np.array(center).shape[0]

    def test(real):
        _real = 1. * real
        for i in range(ndim):
            _real[i] = _real[i] - center[i]
        _shape = _real.shape
        _real.shape = _shape[0], np.product(_shape[1:])

        X = np.dot(_cholinv, _real)
        d = sum(X**2)
        d.shape = _shape[1:]
        value = np.less_equal(d, a)

        del(_real); del(X); del(d)
        gc.collect()
        return value
    return test

def roi_from_array_sampling_coordmap(data, coordmap):
    """
    Return a `CoordinateMapROI` from an array (data) on a coordmap.
    interpolation. Obvious ways to extend this.

    :Parameters:
        data : TODO
            TODO
        coordmap : TODO
            TODO

    :Returns: `CoordinateMapROI`
    """

    if coordmap.shape != data.shape:
        raise ValueError, 'coordmap shape does not agree with data shape'
    voxels = np.nonzero(data)
    coordinate_system = coordmap.output_coordinate_system
    return CoordinateMapROI(coordinate_system, voxels, coordmap)

class ROISequence(list):
    """
    TODO
    """
    pass
