# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Some CoordinateMaps have a domain that are 'array' coordinates,
hence the function of the CoordinateMap can be evaluated at these
'array' points.

This module tries to make these operations easier by defining a class
ArrayCoordMap that is essentially a CoordinateMap and a shape.

This class has two properties: values, transposed_values the
CoordinateMap at np.indices(shape).

The class Grid is meant to take a CoordinateMap and an np.mgrid-like
notation to create an ArrayCoordMap.
"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from .coordinate_map import CoordinateMap, AffineTransform, compose
from .coordinate_map import product as cmap_product
from .coordinate_map import shifted_range_origin
from .coordinate_system import CoordinateSystem


class ArrayCoordMap(object):
    """ Class combining coordinate map and array shape
    
    When the function_domain of a CoordinateMap can be thought of as
    'array' coordinates, i.e. an 'input_shape' makes sense. We can
    than evaluate the CoordinateMap at np.indices(input_shape)
    """
    def __init__(self, coordmap, shape):
        """
        Parameters
        ----------
        coordmap : ``CoordinateMap``
           A CoordinateMap with function_domain that are 'array'
           coordinates.
        shape : sequence of int
           The size of the (implied) underlying array.

        Examples
        --------
        >>> aff = np.diag([0.6,1.1,2.3,1])
        >>> aff[:3,3] = (0.1, 0.2, 0.3)
        >>> cmap = AffineTransform.from_params('ijk', 'xyz', aff)
        >>> cmap.ndims # number of (input, output) dimensions
        (3, 3)
        >>> acmap = ArrayCoordMap(cmap, (1, 2, 3))
        
        Real world values at each array coordinate, one row per array
        coordinate (6 in this case), one column for each output
        dimension (3 in this case)
        
        >>> acmap.values
        array([[ 0.1,  0.2,  0.3],
               [ 0.1,  0.2,  2.6],
               [ 0.1,  0.2,  4.9],
               [ 0.1,  1.3,  0.3],
               [ 0.1,  1.3,  2.6],
               [ 0.1,  1.3,  4.9]])

        Same values, but arranged in np.indices / np.mgrid format, first
        axis is for number of output coordinates (3 in our case), the
        rest are for the input shape (1, 2, 3)

        >>> acmap.transposed_values.shape
        (3, 1, 2, 3)
        >>> acmap.transposed_values
        array([[[[ 0.1,  0.1,  0.1],
                 [ 0.1,  0.1,  0.1]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[ 0.2,  0.2,  0.2],
                 [ 1.3,  1.3,  1.3]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[ 0.3,  2.6,  4.9],
                 [ 0.3,  2.6,  4.9]]]])
        """
        self.coordmap = coordmap
        self.shape = tuple(shape)

    def _evaluate(self, transpose=False):
        """
        If the coordmap has a shape (so that it can be thought of as a
        map from voxels to some output space), return the range of the
        coordmap, i.e. the value at all the voxels.

        Parameters
        ----------
        coordmap : `CoordinateMap`
        transpose : bool, optional
           If False (the default), the result is a 2-dimensional
           ndarray with shape[1] == coordmap.ndims[1]. That is, the
           result is a list of output values.  Otherwise, the shape is
           (coordmap.ndims[1],) + coordmap.shape.

        Returns
        -------
        values : array
           Values of self.coordmap evaluated at np.indices(self.shape).
        """
        indices = np.indices(self.shape).astype(
            self.coordmap.function_domain.coord_dtype)
        tmp_shape = indices.shape
        # reshape indices to be a sequence of coordinates
        indices.shape = (self.coordmap.ndims[0], np.product(self.shape))
        # evaluate using coordinate map mapping
        _range = self.coordmap(indices.T)
        if transpose:
            # reconstruct np.indices format for output
            _range = _range.T
            _range.shape = (_range.shape[0],) + tmp_shape[1:]
        return _range 

    def _getvalues(self):
        return self._evaluate(transpose=False)
    values = property(_getvalues, 
                      doc='Get values of ArrayCoordMap in a '
                      '2-dimensional array of shape '
                      '(product(self.shape), self.coordmap.ndims[1]))')

    def _getindices_values(self):
        return self._evaluate(transpose=True)
    transposed_values = property(_getindices_values, doc='Get values of ArrayCoordMap in an array of shape (self.coordmap.ndims[1],) + self.shape)')

    def __getitem__(self, slicers):
        """
        Return a slice through the coordmap.

        Parameters
        ----------
        slicers : int or tuple
           int, or sequence of any combination of integers, slices.  The
           sequence can also contain one Ellipsis. 
        """
        # slicers might just be just one thing, so convert to tuple
        if type(slicers) != type(()):
            slicers = (slicers,)
        # raise error for anything other than slice, int, Ellipsis
        have_ellipsis = False # check for >1 Ellipsis
        for i in slicers:
            if isinstance(i, np.ndarray):
                raise ValueError('Sorry, we do not support '
                                 'ndarrays (fancy indexing)')
            if i == Ellipsis:
                if have_ellipsis:
                    raise ValueError(
                        "only one Ellipsis (...) allowed in slice")
                have_ellipsis = True
                continue
            try:
                int(i)
            except TypeError:
                if hasattr(i, 'start'): # probably slice
                    continue
                raise ValueError('Expecting int, slice or Ellipsis')
        # allow slicing of form [...,1]
        if have_ellipsis:
            # convert ellipsis to series of slice(None) objects.  For
            # example, if the coordmap is length 3, we convert (...,1)
            # to (slice(None), slice(None), 1) - equivalent to [:,:,1]
            ellipsis_start = list(slicers).index(Ellipsis)
            inds_after_ellipsis = slicers[(ellipsis_start+1):]
            # the ellipsis continues until any remaining slice specification
            n_ellipses = len(self.shape) - ellipsis_start - len(inds_after_ellipsis)
            slicers = (slicers[:ellipsis_start]
                     + n_ellipses * (slice(None),)
                     + inds_after_ellipsis)
        return _slice(self.coordmap, self.shape, *slicers)

    @staticmethod
    def from_shape(coordmap, shape):
        """
        Create an evaluator assuming that coordmap.function_domain
        are 'array' coordinates.

        """
        slices = tuple([slice(0,s,1) for s in shape])
        return Grid(coordmap)[slices]

    def __repr__(self):
        return "ArrayCoordMap(\n  coordmap=" + \
            '\n  '.join(repr(self.coordmap).split('\n')) + ',\n  shape=%s' % repr(self.shape) + '\n)'

def _slice(coordmap, shape, *slices):
    """
    Slice a 'voxel' CoordinateMap's function_domain with slices. A
    'voxel' CoordinateMap is interpreted as a coordmap having a shape.
    """
    
    if len(slices) < coordmap.ndims[0]:
        slices = (list(slices) +
                  [slice(None,None,None)] * (coordmap.ndims[0] - len(slices)))

    ranges = [np.arange(s) for s in shape]
    cmaps = []
    keep_in_output = []

    dtype = coordmap.function_domain.coord_dtype
    newshape = []
    for i, __slice in enumerate(slices):
        ranges[i] = ranges[i][__slice]

        try:
            start = ranges[i][0]
        except IndexError:
            try:
                start = int(ranges[i])
            except TypeError:
                raise ValueError('empty slice for dimension %d, '
                                 'coordinate %s' % 
                                 (i, coordmap.function_domain.coord_names[i]))

        if ranges[i].shape == ():
            step = 0
            start = int(ranges[i])
            l = 1
        elif ranges[i].shape[0] > 1:
            start = ranges[i][0]
            step = ranges[i][1] - ranges[i][0]
            l = ranges[i].shape[0]
            keep_in_output.append(i)
        else:
            start = ranges[i][0]
            step = 0.
            l = 1
            keep_in_output.append(i)

        if step > 1:
            name = coordmap.function_domain.coord_names[i] + '-slice'
        else:
            name = coordmap.function_domain.coord_names[i]
        cmaps.append(AffineTransform(
                CoordinateSystem([name], coord_dtype=dtype),
                CoordinateSystem([coordmap.function_domain.coord_names[i]]),
                np.array([[step, start],[0,1]], dtype=dtype))) 

        if i in keep_in_output:
            newshape.append(l)
    slice_cmap = cmap_product(*cmaps)

    # Identify the origin in the range of cmap
    # with the origin in the domain of coordmap

    slice_cmap = shifted_range_origin(slice_cmap, np.zeros(slice_cmap.ndims[1]),
                                      coordmap.function_domain.name)

    # Reduce the size of the matrix
    innames = slice_cmap.function_domain.coord_names
    inmat = []
    function_domain = CoordinateSystem(
        [innames[i] for i in keep_in_output],
        'input-slice',
        coordmap.function_domain.coord_dtype)
    A = np.zeros((coordmap.ndims[0]+1, len(keep_in_output)+1))
    for j, i in enumerate(keep_in_output):
        A[:,j] = slice_cmap.affine[:,i]
    A[:,-1] = slice_cmap.affine[:,-1]
    A = A.astype(function_domain.coord_dtype)
    slice_cmap = AffineTransform(function_domain, coordmap.function_domain, A)
    return ArrayCoordMap(compose(coordmap, slice_cmap), tuple(newshape))
                   


class Grid(object):
    """
    Simple class to construct AffineTransform instances with slice notation
    like np.ogrid/np.mgrid.

    >>> c = CoordinateSystem('xy', 'input')
    >>> g = Grid(c)
    >>> points = g[-1:1:21j,-2:4:31j]
    >>> points.coordmap.affine
    array([[ 0.1,  0. , -1. ],
           [ 0. ,  0.2, -2. ],
           [ 0. ,  0. ,  1. ]])

    >>> print(points.coordmap.function_domain)
    CoordinateSystem(coord_names=('i0', 'i1'), name='product', coord_dtype=float64)
    >>> print(points.coordmap.function_range)
    CoordinateSystem(coord_names=('x', 'y'), name='input', coord_dtype=float64)

    >>> points.shape
    (21, 31)
    >>> print(points.transposed_values.shape)
    (2, 21, 31)
    >>> print(points.values.shape)
    (651, 2)
    """

    def __init__(self, coords):
        """
        Initialize Grid object

        Parameters
        ----------
        coords: ``CoordinateMap`` or ``CoordinateSystem``
           A coordinate map to be 'sliced' into. If
           coords is a CoordinateSystem, then an
           AffineTransform instance is created with coords
           with identity transformation.
        """

        if isinstance(coords, CoordinateSystem):
            coordmap = AffineTransform.identity(coords.coord_names,
                                                coords.name)
        elif not (isinstance(coords, CoordinateMap) or isinstance(coords, AffineTransform)):
            raise ValueError('expecting either a CoordinateMap, CoordinateSystem or AffineTransform for Grid')
        else:
            coordmap = coords
        self.coordmap = coordmap

    def __getitem__(self, index):
        """
        Create an AffineTransform coordinate map with into self.coords with
        slices created as in np.mgrid/np.ogrid.
        """
        dtype = self.coordmap.function_domain.coord_dtype
        results = [a.ravel().astype(dtype) for a in np.ogrid[index]]
        if len(results) != len(self.coordmap.function_domain.coord_names):
            raise ValueError('the number of slice objects must match ' 
                             'the number of input dimensions')
        cmaps = []
        for i, result in enumerate(results):
            if result.shape[0] > 1:
                step = result[1] - result[0]
            else:
                step = 0
            start = result[0]
            cmaps.append(AffineTransform(
                    CoordinateSystem(['i%d' % i], coord_dtype=dtype),
                    CoordinateSystem([self.coordmap.function_domain.coord_names[i]],
                                     coord_dtype=dtype),
                    np.array([[step, start],[0,1]], dtype=dtype)))

        shape = [result.shape[0] for result in results]
        cmap = cmap_product(*cmaps)

        # Identify the origin in the range of cmap
        # with the origin in the domain of self.coordmap

        cmap = shifted_range_origin(cmap, 
                                    np.zeros(cmap.ndims[1]),
                                    self.coordmap.function_domain.name)

        return ArrayCoordMap(compose(self.coordmap, cmap), tuple(shape))
