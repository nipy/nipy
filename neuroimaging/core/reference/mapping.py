"""
Mappings define a transformation between sets of coordinate systems.

These mappings can be used to transform between voxel space and real space,
for example.
"""

__docformat__ = 'restructuredtext'

import csv, copy
import urllib
from struct import unpack

import numpy as np
from numpy.linalg import inv, pinv

def matvec_from_transform(transform):
    """ Split a tranform into it's matrix and vector components. """
    ndimin = transform.shape[0] - 1
    ndimout = transform.shape[1] - 1
    matrix = transform[0:ndimin, 0:ndimout]
    vector = transform[0:ndimin, ndimout]
    return matrix, vector

def transform_from_matvec(matrix, vector):
    """ Combine a matrix and vector into a transform. """
    nin, nout = matrix.shape
    t = np.zeros((nin+1,nout+1), matrix.dtype)
    t[0:nin, 0:nout] = matrix
    t[nin,   nout] = 1.
    t[0:nin, nout] = vector
    return t

class Mapping(object):
    """
    A generic mapping class that allows composition, inverses, etc. A mapping
    needs only input and output coordinates and a transform between the two,
    and an optional inverse.
    """

    def __init__(self, map, inverse=None):
        """
        :Parameters:
            map : callable
                A function which takes a coordinate vector and returns a new coordinate vector
            inverse : callable
                An inverse function to ``map``
            ndim : (``int``, ``int``)
                The number of (input, output) dimensions of the mapping.
        """
        self._map = map
        if inverse is not None:
            self._inverse = inverse
        
    def __call__(self, x):
        """ Apply this mapping to the given coordinates. 
        It expects a coordinate array of shape (*, self.transform.shape[1]-1)
        as it multiplies x on the right by the transformation matrix.
        
        :Parameters:
            x : ``numpy.ndarray``
                Points at which to evaluate the mapping.
        
        :Returns: ``numpy.ndarray``
        """
        self._checkshape(x)
        return self._map(x)

    def _checkshape(self, x):
        """
        Verify that x has the proper shape for evaluating the mapping
        """
        if x.dtype.isbuiltin:
            if x.ndim > 2:
                raise ValueError('if dtype is builtin, expecting a 2-d with shape[-1] == ndim_input, or a 1-d array of shape[0] == ndim_input')
        elif x.ndim > 1:
            raise ValueError, 'if dtype is not builtin, expecting 1-d array, or a 0-d array' 

    def _getisinvertible(self):
        """
        Does this mapping have an inverse?

        :Returns: ``bool``
        """
        if self.inverse is not None:
            return callable(self.inverse)
        return False

    isinvertible = property(_getisinvertible)

    def _getinverse(self):
        """
        Create a new Mapping instance which is the inverse of self.

        :Returns: `Mapping`
        """
        try:
            return Mapping(self._inverse, self)
        except AttributeError:
            return None
    inverse = property(_getinverse)


class Affine(Mapping):
    """
    A class representing an affine transformation in n axes.
    
    This class adds a transform member, which is a matrix representing
    the affine transformation. This matrix is used to perform mappings,
    rather than having an explicit mapping function. 
    """

    @staticmethod
    def identity(ndim=3):
        """ Return an identity affine transformation.

        :Returns: `Affine`
        """
        return Affine(np.identity(ndim+1))

    def __eq__(self, other):
        if hasattr(other, 'transform'):
            return np.allclose(self.transform, other.transform)
        return False

    def __init__(self, affine_transform):
        """
        :Parameters:
            affine_transform : ``numpy.ndarray``
                A transformation matrix

        """
        self.transform = affine_transform
        ndimin = affine_transform.shape[0] - 1
        ndimout = affine_transform.shape[1] - 1
        self._fmatrix, self._fvector = matvec_from_transform(affine_transform)
        Mapping.__init__(self, None)

    def _checkshape(self, x):
        """
        Verify that x has the proper shape for evaluating the mapping
        """
        A, b = self.params
        ndim = A.shape[::-1]
        if x.dtype.isbuiltin:
            if x.ndim > 2 or x.shape[-1] != ndim[0]:
                raise ValueError('if dtype is builtin, expecting a 2-d array of shape (*,%d) or a 1-d array of shape (%d,)' % (ndim[0], ndim[0]))
        elif x.ndim > 1:
            raise ValueError, 'if dtype is not builtin, expecting 1-d array, or a 0-d array' 

    def get_params(self):
        return matvec_from_transform(self.transform)
    params = property(get_params,doc='Matrix, vector representation of affine')

    def reshape(self, x):
        """
        Typecast x to have the same dtype as self.transform
        and return a ravel'ed version with appropriate
        shape.
        
        If x.dtype is not a built in, it reshapes
        x.ravel().view(self.transform.dtype).

        If x.dtype is builtin it reshapes
        x.ravel(). 
        """

        A, b = self.params
        ndim = A.shape[::-1]

        if x.dtype.isbuiltin:
            y = x.ravel() # do nothing except ravel()
        else:
            y = x.ravel().view(self.transform.dtype)
        y.shape = (y.shape[0] / ndim[0], ndim[0])
        return y

    def __call__(self, x):
        """ Apply this mapping to the given coordinates. 
        It expects a coordinate array of shape (*, self.transform.shape[1]-1)
        as it multiplies x on the right by the transformation matrix.
        
        :Parameters:
            x : ``numpy.ndarray``
                Points at which to evaluate the mapping.
        
        :Returns: ``numpy.ndarray``
        """

        x = np.asarray(x)
        self._checkshape(x)

        A, b = self.params
        tx = self.reshape(x)
        value = np.dot(tx, A.T) 
        value += np.multiply.outer(np.ones(value.shape[0]), b)
        if x.shape == ():
            value.shape == ()
        return value

    def _getinverse(self):
        """
        Create a new `Affine` instance which is the inverse of self.

        :Returns: `Affine`
        """
        # Does it make sense to return pinv?
        try:
            return Affine(inv(self.transform))
        except np.linalg.linalg.LinAlgError:
            return None

    inverse = property(_getinverse)

def compose(*mappings):
    """ Mapping composition

    :Parameters:
         mappings: sequence of Mappings
                The mapping to compose with.
    :Returns: `Mapping`

    :Note: If all mappings are Affine, return an Affine instance.

    """
    def _compose2(map1, map2):
        forward = lambda input: map1(map2(input))
        if map1.isinvertible and map2.isinvertible:
            backward = lambda output: map2.inverse(map1.inverse(output))
        else:
            backward = None
        return Mapping(forward, inverse=backward)
    mapping = mappings[-1]
    for i in range(len(mappings)-2,-1,-1):
        mapping = _compose2(mappings[i], mapping)

    notaffine = filter(lambda mapping: not isinstance(mapping, Affine), mappings)
    if not notaffine:
        mapping = linearize(mapping, mappings[-1].transform.shape[1]-1)
    return mapping
    
def linearize(mapping, ndimin, step=np.array(1.), origin=None):
    """
    Given a Mapping of ndimin variables, 
    with an input builtin dtype, return the linearization
    of mapping at origin based on a given step size
    in each coordinate axis.

    If not specified, origin defaults to np.zeros(ndimin, dtype=dtype).
    
    :Inputs: 
        mapping: ``Mapping``
              A Mapping to linearize
        ndimin: ``int``
              Number of input dimensions to mapping
        origin: ``ndarray``
              Origin at which to linearize mapping
        step: ``ndarray``
              Step size, an ndarray with step.shape == ().

    :Returns:
        A: ``ndarray``
            Linearization of mapping in homogeneous coordinates, i.e. 
            an array of size (ndimout+1, ndimin+1) where
            ndimout = mapping(origin).shape[0].

    :Notes: The dtype of the resulting Affine mapping
            will be the dtype of mapping(origin)/step, regardless
            of the input dtype.

    """
    step = np.asarray(step)
    if step.shape != ():
        raise ValueError('step should be a scalar value')
    if origin is None:
        origin = np.zeros(ndimin, step.dtype)
    else:
        if origin.dtype != step.dtype:
            warnings.warn('origin.dtype != step.dtype in function linearize, using step.dtype')
        origin = np.asarray(origin, dtype=step.dtype)
        if origin.shape != (ndimin,):
            raise ValueError('origin.shape != (%d,)' % ndimin)
    b = mapping(origin)

    origin = np.multiply.outer(np.ones(ndimin), origin)
    y1 = mapping(step*np.identity(ndimin) + origin)
    y0 = mapping(origin)

    ndimout = y1.shape[1]
    C = np.zeros((ndimout+1, ndimin+1), (y0/step).dtype)
    C[-1,-1] = 1
    C[:ndimout,-1] = b
    C[:ndimout,:ndimin] = (y1 - y0).T / step
    return Affine(C)

