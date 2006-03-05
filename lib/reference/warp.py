import csv, string
import numpy as N
from numpy.linalg import inverse
from numpy.random import standard_normal
import coordinate_system, axis
import string, sets, StringIO, urllib, re, struct
import enthought.traits as traits

def _2matvec(transform):
    ndim = transform.shape[0] - 1
    matrix = transform[0:ndim,0:ndim]
    vector = transform[0:ndim,ndim]
    return matrix, vector

def tofile(warp, filename):
    if not isinstance(warp, Affine):
        raise NotImplementedError, 'only Affine transformations can be written out'

    t = warp.transform

    matfile = file(filename, 'w')
    writer = csv.writer(matfile, delimiter='\t')
    for row in t:
        writer.writerow(row)
    matfile.close()
    return

def fromfile(filename, names=axis.space, input='voxel', output='world', delimiter='\t'):
    """
    Read in an affine transformation matrix and return an instance of Affine
    with named axes and input and output coordinate systems.

    For now, the format is assumed to be a tab-delimited file.
    Other formats should be added.
    """
    matfile = file(filename)
    t = []
    reader = csv.reader(matfile, delimiter=delimiter)
    for row in reader:
        t.append(map(string.atof, row))
    t = N.array(t)
    matfile.close()
    return frommatrix(t, names=names, input=input, output=output)

def isdiagonal(matrix, tol=1.0e-7):
    ndim = matrix.shape[0]
    D = N.diag(N.diagonal(matrix))
    dmatrix = matrix - D
    dmatrix.shape = (ndim,)*2
    dnorm = N.add.reduce(dmatrix**2)
    fmatrix = 1. * matrix
    fmatrix.shape = dmatrix.shape
    norm = N.add.reduce(fmatrix**2)
    if N.add.reduce(dnorm / norm) < tol:
        return True
    else:
        return False

def frommatrix(matrix, names=axis.space, input='voxel', output='world'):
    """
    Return an Affine instance with named axes and input and output coordinate systems.
    """
    ndim = matrix.shape[0] - 1

    inaxes = []
    outaxes = []
    for i in range(ndim):
        inaxes.append(axis.Axis(name=names[i]))
        outaxes.append(axis.Axis(name=names[i]))
    incoords = coordinate_system.CoordinateSystem('input', inaxes)
    outcoords = coordinate_system.CoordinateSystem('input', inaxes)
    return Affine(incoords, outcoords, matrix)

def IdentityWarp(ndim=3, names=axis.space, input='voxel', output='world'):
    """
    Identity Affine transformation.
    """
    return frommatrix(N.identity(ndim+1), names=names, input=input, output=output)

class Warp(traits.HasTraits):
    """
    A generic warp class that allows composition, inverses, etc. A warp needs only input and output coordinates and a transform between the two, and an optional inverse.
    """

    maptype = traits.String('generic')
    name = traits.String('transform')
    input_coords = traits.Any()
    output_coords = traits.Any()
    _map = traits.Any()

    def __str__(self):
        value = '%s:%s=%s\n' % (str(self.name), 'input', str(self.input_coords))
        value = value + '%s:%s=%s\n' % (str(self.name), 'output', str(self.output_coords))
        value = value + '%s:%s=%s\n' % (str(self.name), 'map', str(self._map))
        if hasattr(self, '_inverse'):
            value = value + '%s:%s=%s\n' % (str(self.name), 'map', str(self._inverse))
        return value

    def __eq__(self, other):
        test = (self.input_coords == other.input_coords) * (self.output_coords == other.output_coords) * (self._map == other._map)
        return bool(test)

    def __init__(self, input_coords, output_coords, _map, _inverse=None, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.input_coords = input_coords
        self.output_coords = output_coords
        self.ndim = self.input_coords.ndim
        self._map = _map
        self._inverse = _inverse

    def __call__(self, x, inverse=False):
        return self.map(x, inverse=inverse)

    def inverse(self):
        """
        Return the inverse Warp.
        """

        if hasattr(self, '_inverse'):
            return Warp(self.output_coords, self.input_coords, self._inverse, self.map, maptype=self.maptype)
        else:
            raise ValueError, 'non-invertible warp.'
         
   
    def map(self, coords, inverse=False):
        if not inverse:
            return self._map(coords)
        else:
            return self._inverse(coords)

    def __mul__(self, other):
        """
        If this method is not over-written we get complaints about sequences.
        """
        return other.__rmul__(self)
    
    def __rmul__(self, other):
#        if self.output_coords.name.strip() != other.input_coords.name.strip():
#            raise ValueError, 'input and output coordinate names do not match.'
        def _map(coords):
            return other.map(self.map(coords))
        if hasattr(self, '_inverse') and hasattr(other, '_inverse'):
            def _inverse(coords):
                return self.map(other.map(coords, inverse=True), inverse=True)
        else:
            _inverse = None
        return Warp(self.input_coords, other.output_coords, _map, _inverse=_inverse
)
    def reslice(self, which, inname=None, outname=None, sort=True):
        """
        Reorder and/or subset a warp, uses subset of input_coords.axes to determine subset.

        Warning: this does not know about the \'new\' CoordinateSystem classes.

        """

        dimnames = list(sets.Set(self.input_coords.dimnames).difference(which.keys()))
        order = [self.input_coords.dimnames.index(dimname) for dimname in dimnames]
        if sort:
            order.sort() # keep order of coordinates: a good idea or not?

        whichmap = {}
        for dimname in which.keys():
            whichmap[dimname] = self.input_coords.dimnames.index(dimname)
        
        indim = [self.input_coords.axes[i] for i in order]
        if inname is None:
            inname = 'voxel'
        incoords = coordinate_system.CoordinateSystem(inname, indim)
        
        outdim = [self.output_coords.axes[i] for i in order]
        if outname is None:
            outname = 'world'
        outcoords = coordinate_system.CoordinateSystem(outname, outdim)

        def _map(voxel, _map=whichmap, which=which, ndim=self.ndim, order=order):
            try:
                _shape = voxel.shape[1:]
            except:
                _shape = ()
            _voxel = N.zeros((ndim,) + _shape, N.Float)
            for dimname in which.keys():
                _voxel[_map[dimname]] = which[dimname]
            for i in range(len(order)):
                _voxel[order[i]] = voxel[i]

            _value = self._map(_voxel)
            _value = array([_value[i] for i in order])
            return _value

        return Warp(incoords, outcoords, _map, _inverse=None) 

class Affine(Warp):
    """
    A class representing an affine transformation in n axes.
    """

    def __init__(self, input_coords, output_coords, transform, name='transform'):
        self.name = name
        self.input_coords = input_coords
        self.output_coords = output_coords
        self.transform = transform
        self.ndim = transform.shape[0] - 1
        self.fmatrix, self.fvector = _2matvec(transform)
        self.bmatrix, self.bvector = _2matvec(inverse(transform))
        def _map(coords):
            return N.dot(self.fmatrix, coords) + self.fvector
        def _inverse(coords):
            return N.dot(self.bmatrix, coords) + self.bvector
        Warp.__init__(self, input_coords, output_coords, _map, _inverse=_inverse, maptype='affine')

    def __ne__(self, other):
        return not self.__eq__(other)

    def subset(self, dimnames, inname=None, outname=None, sort=True):
        """
        Reorder and/or subset a warp, uses subset of output_coords.axes to determine subset.

        Warning: this does not know about the \'new\' Coordinate classes.

        """

        order = [self.output_coords.dimnames.index(dimname) for dimname in dimnames]
        if sort:
            order.sort() # keep order of coordinates: a good idea or not?

        indim = [self.input_coords.axes[i] for i in order]
        if inname is None:
            inname = 'voxel'
        incoords = coordinates_system.CoordinateSystem(inname, indim)
        
        outdim = [self.output_coords.axes[i] for i in order]

        if outname is None:
            outname = 'world'
        outcoords = coordinate_system.CoordinateSystem(outname, outdim)

        l = len(order)
        transform = N.zeros((l+1,)*2, N.Float)
        transform[l,l] = 1.0
        for i in range(l):
            for j in range(l):
                transform[i,j] = self.transform[order[i],order[j]]
            transform[i,l] = self.transform[order[i],self.ndim]
                
        return Affine(incoords, outcoords, transform, shape=_shape)

    def __eq__(self, other):
        try:
            if tuple(self.transform.flat) != tuple(other.transform.flat):
                return False
        except:
            return False
        
        return True

    def map(self, coords, inverse=False, alpha=1.0):
        if not inverse:
            value = N.dot(self.fmatrix, coords)
            if len(value.shape) > 1:
                value = value + N.multiply.outer(self.fvector, N.ones(value.shape[1]))
            elif alpha != 0.0:
                value = value + self.fvector * alpha # for derivatives, we don't want translation...
        else:
            value = N.dot(self.bmatrix, coords)
            if len(value.shape) > 1:
                value = value + multiply.outer(self.bvector, N.ones(value.shape[1:]))
            elif alpha != 0.0:
                value = value + self.bvector * alpha # for derivatives, we don't want translation...
        return value
    
    def inverse(self):
        _transform = inverse(self.transform)
        return Affine(self.output_coords, self.input_coords, _transform)

    def __str__(self):
        value = '%s:%s=%s\n' % (str(self.name), 'input', str(self.input_coords))
        value = value + '%s:%s=%s\n' % (str(self.name), 'output', str(self.output_coords))
        value = value + '%s:%s=%s\n' % (str(self.name), 'fmatrix', str(self.fmatrix))
        value = value + '%s:%s=%s\n' % (str(self.name), 'fvector', str(self.fvector))
        return value

    def __rmul__(self, other):
        if other.maptype == 'affine':
##            if self.output_coords.name.strip() != other.input_coords.name.strip():
##                raise ValueError, 'input and output coordinate names do not match.'

            return Affine(self.input_coords, other.output_coords, N.dot(other.transform, self.transform))
        else:
            return Warp.__rmul__(self, other)

    def __str__(self):
        value = '%s:input=%s\n%s:output=%s\n%s:fmatrix=%s\n%s:fvector=%s' % (self.name, self.input_coords.name, self.name, self.output_coords.name, self.name, `self.fmatrix`, self.name, `self.fvector`)
        return value
    
    def translate_input(self, x = -N.ones((3,))):
        """Modify (in place) a affine transform so that its input is translated by x, which defaults to -1 for handling C->matlab indexing."""
        ndim = self.ndim
        self.transform = N.dot(self.transform, inverse(_translation_transform(x, self.ndim)))
        for i in range(ndim):
            self.input_coords.axes[i].start = self.input_coords.axes[i].start - x[i]
        self.fmatrix, self.fvector = _2matvec(self.transform)
        self.bmatrix, self.bvector = _2matvec(inverse(self.transform))

    def translate_output(self, x = -N.ones((3,))):
        """Modify (in place) an affine transform so that its output is translated by x, which defaults to -1 for handling C->matlab indexing"""
        ndim = self.ndim
        self.transform = N.dot(_translation_transform(x, self.ndim), self.transform)
        for i in range(ndim):
            self.output_coords.axes[i].start = self.output_coords.axes[i].start + x[i]
        self.fmatrix, self.fvector = _2matvec(self.transform)
        self.bmatrix, self.bvector = _2matvec(inverse(self.transform))


def permutation_matrix(order=range(3)[2::-1]):
    """Create an NxN permutation matrix from a sequence, containing the values 0,...,N-1."""
    n = len(order)
    matrix = N.zeros((n,n))
    if sets.Set(order) != sets.Set(range(n)):
        raise ValueError, 'order should be a sequence of integers with values, 0 ... len(order)-1.'
    for i in range(n):
        matrix[i,order[i]] = 1
    return matrix

def permutation_transform(order=range(3)[2::-1]):
    """Create an (N+1)x(N+1) permutation transformation from a sequence, containing the values 0,...,N-1."""
    ndim = len(order)
    ptransform = N.zeros((ndim+1,ndim+1), N.Float)
    ptransform[0:ndim,0:ndim] = permutation_matrix(order=order)
    ptransform[ndim,ndim] = 1.

    return ptransform

def _translation_transform(x, ndim):
    """Create a matrix representing translation by x."""
    _transform = N.identity(ndim+1)
    _transform[0:ndim,ndim] = _transform[0:ndim,ndim] + x 
    return _transform

def linearize(warp, seed=None):
    d = warp.ndim
    if seed is None:
        seed = N.zeros(d, N.Float)
    shift = warp(seed)
    A = N.zeros((d,)*2, N.Float)

    for i in range(d):
        dx = N.zeros(d, N.Float)
        dx[i] = 1.
        A[i] = warp(seed + dx) - warp(seed)

    transform = N.zeros((warp.ndim+1,)*2, N.Float)
    transform[0:warp.ndim,0:warp.ndim] = A
    transform[0:warp.ndim,warp.ndim] = shift
    transform[warp.ndim,warp.ndim] = 1.
    w = Affine(warp.input_coords, warp.output_coords, transform)
    return w

def tovoxel(real, warp):
    """Given a warp and a real coordinate, where warp.input_coords are assumed to be voxels, return the closest voxel for real. Will choke if warp is not invertible."""
    _shape = real.shape
    real.shape = (_shape[0], product(_shape[1:]))
    voxel = N.around(warp.map(real, inverse=True))
    real.shape = _shape
    voxel.shape = _shape
    return N.array(voxel)

def matlab2python(warp):
    """
    Take that maps matlab voxels to (matlab-ordered) world coordinates and make it python-oriented. This means that if warp(v_x,v_y,v_z)=(w_x,w_y,w_z) then the return will send (v_z,v_y,v_x) to (w_z,w_y,w_x).
    """

    ndim = warp.input_coords.ndim
    t1 = N.zeros((ndim+1,)*2, N.Float)
    t1[0:ndim,0:ndim] = permutation_matrix(range(ndim)[::-1])
    t1[ndim, ndim] = 1.0

    t2 = 1. * t1
    t1[0:ndim,ndim] = 1.0

    n = warp.ndim
    d1 = [warp.input_coords.axes[n-1-i] for i in range(n)]
    in1 = coordinate_system.CoordinateSystem(warp.input_coords.name, d1)
    w1 = Affine(in1, warp.input_coords, t1)
    
    d2 = [warp.output_coords.axes[n-1-i] for i in range(n)]
    out2 = coordinate_system.CoordinateSystem(warp.output_coords.name, d2)
    w2 = Affine(warp.output_coords, out2, t2)

    w = (w2 * warp) * w1
    return w

fortran2C = matlab2python

def python2matlab(warp):
    """
    Inverse of matlab2python -- see this function for help.
    """

    ndim = warp.input_coords.ndim
    t1 = N.zeros((ndim+1,)*2, N.Float)
    t1[0:ndim,0:ndim] = permutation_matrix(range(ndim)[::-1])
    t1[ndim, ndim] = 1.0

    t2 = 1. * t1
    t1[0:ndim,ndim] = -1.0

    n = warp.ndim
    d1 = [warp.input_coords.axes[n-1-i] for i in range(n)]
    in1 = coordinate_system.CoordinateSystem(warp.input_coords.name, d1)
    w1 = Affine(in1, warp.input_coords, t1)
    
    d2 = [warp.output_coords.axes[n-1-i] for i in range(n)]
    out2 = coordinate_system.CoordinateSystem(warp.output_coords.name, d2)
    w2 = Affine(warp.output_coords, out2, t2)

    w = (w2 * warp) * w1

    return w

C2fortran = python2matlab

MNI_warp = Affine(coordinate_system.MNI_voxel, coordinate_system.MNI_world, coordinate_system.MNI_world.transform())
MNI_warp([36,63,45])
