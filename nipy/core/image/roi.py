# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Template region of interest (ROI) module
"""

__docformat__ = 'restructuredtext'

# FIXME: This module needs some attention. There are no unit tests for it
# so it's hard to say whether it works correctly or not.

"""
* What role does the ROI.coordinate_system play?  It does not seem to be used
* The DiscreteROI seems to assume that the ``voxels`` are mm (real value)
  points, but the CoordinateMapROI assumes (by indexing the image) that the
  voxels are i,j,k indices
* The coordmap no longer has a .shape attribute, so can't support the mask
  method in CoordinateMapROI any more.
"""
import warnings

warnings.warn('The ROI module is not stable, and is probably broken')

import gc

import numpy as np

class ROI(object):
    """
    This is the basic ROI class, which we model as basically
    a function defined on Euclidean space, i.e. R^3. For practical
    purposes, this function is evaluated on the range of a Mapping
    instance.
    """

    def __init__(self, coordinate_system):
        """ Initialize ROI instance

        Parameters
        ----------
        coordinate_system : ``CoordinateSystem`` instance
        """
        self.coordinate_system = coordinate_system


class ContinuousROI(ROI):
    """
    Create an `ROI` with a binary function in a given coordinate system.
    """
    def __init__(self, coordinate_system, bfn, args=None, ndim=3):
        """ Initialize continuous ROI instance

        Parameters
        ----------
        coordinate_system : ``CoordinateSystem`` instance
            TODO
        bfn : callable
            binary function accepting real-value points as input, and any args
            in `args`, returning 1 at points inside the ROI and 0 for points
            outside the ROI.
        args : sequence
            arguments to be passed to `bfn` other then real-valued points
        ndim : int
            number of dimensions.
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
        """ Call binary function on points in `real`
        Parameters
        ----------
        real : array shape (N, ndim)
            Array defining N points

        Returns
        -------
        inside_tf : array shape (N,)
            Array corresponding to `real` where 0 means that point was outside
            the ROI, and 1 means it was inside the ROI.
        """
        return np.not_equal(self.bfn(real, **self.args), 0)

    def todiscrete(self, voxels):
        """ Return a `DiscreteROI` instance at the `voxels` in the ROI.

        Parameters
        ----------
        voxels : array shape (N, 3)
            voxel points in real space

        Returns
        -------
        droi : ``DiscreteROI`` instance
            discrete ROI where roi defined by voxels inside `self`
        """
        v = []
        for voxel in voxels:
            if self(voxel):
                v.append(voxel)
        return DiscreteROI(self.coordinate_system, v)


class DiscreteROI(ROI):
    """
    ROI defined from discrete points
    """
    def __init__(self, coordinate_system, voxels):
        """ Initialize discrete ROI

        Parameters
        ----------
        coordinate_system : TODO
            TODO
        voxels : sequence
        """
        ROI.__init__(self, coordinate_system)
        self.voxels = set(voxels)

    def __iter__(self):
        """ Return iterator
        """
        self.voxels = iter(self.voxels)
        return self

    def next(self):
        """ Return next point in ROI
        """
        return self.voxels.next()

    def pool(self, fn, **extra):
        """
        Pool data from an image over the ROI -- return fn evaluated at
        each voxel.

        Parameters
        ----------
        fn : callable
            function to apply to each voxel
        \\**extras : kwargs
            keyword arguments to pass to `fn`

        Returns
        -------
        proc_pts : list
            result of `fn` applied to each point within ROI
        """
        return [fn(voxel, **extra) for voxel in self.voxels]

    def feature(self, fn, **extra):
        """ Return a feature of an image within the ROI.

        Take the mean of voxel (point) features in ROI.

        Parameters
        ----------
        fn : callable
            accepts point and kwargs \\**extra, returns value for that point
            (see ``pool`` method)
        \\**extra : kwargs
            keyword arguments to pass to `fn`

        Returns
        -------
        val : object
            result of ``np.mean`` when applied to the values output from `fn`
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
        """ Initialize coordinate map ROI instance

        Parameters
        ----------
        coordinate_system : TODO
            TODO
        voxels : TODO
            TODO
        coordmap : TODO
            TODO
        """
        DiscreteROI.__init__(self, coordinate_system, voxels)
        self.coordmap = coordmap
        # we assume that voxels are (i,j,k) indices - see ``pool`` method

    def pool(self, image):
        """ Pool data from an image over the ROI

        Return image value for each voxel in ROI

        Parameters
        ----------
        image : `image.Image`
            or something with a ``get_data`` method

        Returns
        -------
        vals : list
            values in `image` at voxel points given by ``self.voxels``

        Raises
        ------
        ValueError: if coordinate maps of image and ROI do not match
        """
        if image.coordmap != self.coordmap:
            raise ValueError(
              'to pool an image over a CoordinateMapROI the coordmaps must agree')
        tmp = image.get_data()
        v = [tmp[voxel] for voxel in self.voxels]

        del(tmp); gc.collect()
        return v

    def __mul__(self, other):
        """ Union of two coordinate map ROIs

        Parameters
        ----------
        other : ``CoordinateMapROI`` instance
            roi to combine with ``self``

        Returns
        -------
        union_roi : ``CoordinateMapROI`` instance
            ROI that is the union of points between ``self`` and `other`

        Raises
        ------
        ValueError: if coordinate maps of ``self`` and `other` do not agree
        NotImplementedError: if `other` is not a ``CoordinateMapROI``
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

    def mask(self, img):
        """ Return image with ones within ROI, zeros elsewhere

        :Returns: ``numpy.ndarray`
        """
        raise NotImplementedError('The coordmap interface has changed')
        m = np.zeros(self.coordmap.shape, np.int32)
        for v in self.voxels:
            m[v] = 1.
        return m


def roi_sphere_fn(center, radius):
    """ Binary function for sphere with `center` and `radius`

    Parameters
    center : sequence
        real coordinates point for sphere center
    radius : float
        sphere radius

    Returns
    -------
    sph_fn : function
        binary function accepting points as input, return True if point is
        within sphere, False otherwise
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

    Parameters
    ----------
    form : TODO
        TODO
    a : float
        TODO

    Returns
    -------
    ellipse_fn : function
        binary function of point, returning True if point is within ellipse,
        False otherwise
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

    Parameters
    ----------
    data : array
        Non-zero values in `data` define points in ROI
    coordmap : ``CoordinateMap`` instance
        coordinate map defining relationship of ijk(etc) indices in `data` and
        point space

    Returns
    -------
    cm_roi : ``CoordinateMapROI``
    """
    voxels = np.nonzero(data)
    coordinate_system = coordmap.output_coordinate_system
    return CoordinateMapROI(coordinate_system, voxels, coordmap)
