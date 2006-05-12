import csv, string
import numpy as N
from numpy.linalg import inv as inverse
from numpy.random import standard_normal
from coordinate_system import CoordinateSystem, MNI_voxel, MNI_world
from axis import Axis, space
import string, sets, StringIO, urllib, re, struct
import enthought.traits as traits

def _2matvec(transform):
    ndim = transform.shape[0] - 1
    matrix = transform[0:ndim,0:ndim]
    vector = transform[0:ndim,ndim]
    return matrix, vector

def tofile(mapping, filename):
    if not isinstance(mapping, Affine):
        raise NotImplementedError, 'only Affine transformations can be written out'

    t = mapping.transform

    matfile = file(filename, 'w')
    writer = csv.writer(matfile, delimiter='\t')
    for row in t:
        writer.writerow(row)
    matfile.close()
    return

def matfromfile(infile, delimiter="\t"):
    "Read in an affine transformation matrix from a csv file."
    if type(infile)==types.StringType: infile = file(infile)
    reader = csv.reader(infile, delimiter=delimiter)
    t = N.array([map(string.atof, row) for row in reader])
    infile.close()

def fromfile(infile, names=space, input='voxel', output='world', delimiter='\t'):
    """
    Read in an affine transformation matrix and return an instance of Affine
    with named axes and input and output coordinate systems.

    For now, the format is assumed to be a tab-delimited file.
    Other formats should be added.
    """
    t = matfromfile(infile, delimiter=delimiter)
    return frommatrix(t, names=names, input=input, output=output)

def matfromstr(tstr, ndim=3, delimiter=None):
    """Read a (ndim+1)x(ndim+1) transform matrix from a string."""

    if tstr[0:24] == "mat file created by perl":
        return frombin(tstr)
    else:
        if delimiter is None:
            transform  = N.array(map(string.atof, string.split(tstr)))
        else:
            transform  = N.array(map(string.atof, string.split(tstr, delimiter)))
        transform.shape = (ndim+1,)*2
        return transform

def xfmfromstr(tstr, ndim=3):
    """Read a (ndim+1)x(ndim+1) transform matrix from a string."""

    tstr = tstr.split('\n')
    more = True
    data = []
    outdata = []
    for i in range(len(tstr)):

        if tstr[i].find('/matrix') >= 0:
            for j in range((ndim+1)**2):
                data.append(string.atof(tstr[i+j+1]))

        if tstr[i].find('/outputusermatrix') >= 0:
            for j in range((ndim+1)**2):
                outdata.append(string.atof(tstr[i+j+1]))

    data = N.array(data)
    data.shape = (ndim+1,)*2

    outdata = N.array(outdata)
    outdata.shape = (ndim+1,)*2
    
    return data, outdata


def fromurl(turl, ndim=3):
    """Read a (ndim+1)x(ndim+1) transform matrix from a URL -- tries to autodetect '.mat' and '.xfm'.

    >>> from numarray import *
    >>> x = fromurl('http://kff.stanford.edu/BrainSTAT/fiac3_fonc1.txt')
    >>> y = fromurl('http://kff.stanford.edu/BrainSTAT/fiac3_fonc1_0089.mat')
    >>> print bool(max(abs((x - y).flat) < 1.0e-05))
    True
"""
    
    urlpipe = urllib.urlopen(turl)
    data = urlpipe.read()
    if turl[-3:] == 'mat':
        return matfromstr(data, ndim=ndim)
    elif turl[-3:] == 'xfm':
        return xfmfromstr(data, ndim=ndim)

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

def frommatrix(matrix, names=space, input='voxel', output='world'):
    """
    Return an Affine instance with named axes and input and output coordinate systems.
    """
    axes = [Axis(name=n) for n in names]
    return Affine(
      CoordinateSystem(input, axes), CoordinateSystem(output, axes),
      matrix)

def IdentityMapping(ndim=3, names=space, input='voxel', output='world'):
    """
    Identity Affine transformation.
    """
    return frommatrix(N.identity(ndim+1), names=names, input=input, output=output)

class Mapping(traits.HasTraits):
    """
    A generic mapping class that allows composition, inverses, etc. A mapping needs only input and output coordinates and a transform between the two, and an optional inverse.
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
        Return the inverse Mapping.
        """

        if hasattr(self, '_inverse'):
            return Mapping(self.output_coords, self.input_coords, self._inverse, self.map, maptype=self.maptype)
        else:
            raise ValueError, 'non-invertible mapping.'
   
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
        return Mapping(self.input_coords, other.output_coords, _map, _inverse=_inverse
)
    def reslice(self, which, inname=None, outname=None, sort=True):
        """
        Reorder and/or subset a mapping, uses subset of input_coords.axes to determine subset.

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
        incoords = CoordinateSystem(inname, indim)
        
        outdim = [self.output_coords.axes[i] for i in order]
        if outname is None:
            outname = 'world'
        outcoords = CoordinateSystem(outname, outdim)

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

        return Mapping(incoords, outcoords, _map, _inverse=None) 

class Affine(Mapping):
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
        Mapping.__init__(self, input_coords, output_coords, _map, _inverse=_inverse, maptype='affine')

    def __ne__(self, other):
        return not self.__eq__(other)

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
                value = value + N.multiply.outer(self.bvector, N.ones(value.shape[1:]))
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
            try:
                return Affine(self.input_coords, other.output_coords, N.dot(other.transform, self.transform))
            except:
                fmatrix = N.dot(other.fmatrix, self.fmatrix)
                fvector = N.dot(other.fmatrix, self.fvector) + other.fvector
                return DegenerateAffine(self.input_coords, other.output_coords, fmatrix, fvector)
        else:
            return Mapping.__rmul__(self, other)

    def __str__(self):
        value = '%s:input=%s\n%s:output=%s\n%s:fmatrix=%s\n%s:fvector=%s' % (self.name, self.input_coords.name, self.name, self.output_coords.name, self.name, `self.fmatrix`, self.name, `self.fvector`)
        return value
    
class DegenerateAffine(Affine):
    """
    A subclass of affine with no inverse, i.e. where the map is non-invertible.
    """
    
    nout = traits.Int(3)
    nin = traits.Int(3)

    def __init__(self, input_coords, output_coords, fmatrix, fvector,
                 name='transform'):
        self.name = name
        self.input_coords = input_coords
        self.output_coords = output_coords
        self.fmatrix = fmatrix
        self.fvector = fvector
        self.nin = fmatrix.shape[1]
        self.nout = fmatrix.shape[0]

        def _map(coords):
            return N.dot(self.fmatrix, coords) + self.fvector

        try:
            t = N.zeros((self.ndin+1,)*2, N.Float)
            t[0:self.nin,0:self.nin] = self.fmatrix
            t[self.nin,self.nin] = 1.
            t[0:self.nin,self.nin] = self.fvector
            x = inverse(t)
            Affine.__init__(self, input_coords, output_coords, t, name=name)
        except:
            Mapping.__init__(self, input_coords, output_coords, _map, maptype='affine')


def permutation_matrix(order=range(3)[2::-1]):
    """
    Create an NxN permutation matrix from a sequence, containing the values 0,...,N-1.
    """
    n = len(order)
    matrix = N.zeros((n,n))
    if sets.Set(order) != sets.Set(range(n)):
        raise ValueError, 'order should be a sequence of integers with values, 0 ... len(order)-1.'
    for i in range(n):
        matrix[i,order[i]] = 1
    return matrix

def permutation_transform(order=range(3)[2::-1]):
    """
    Create an (N+1)x(N+1) permutation transformation matrix from a sequence, containing the values 0,...,N-1.
    """
    ndim = len(order)
    ptransform = N.zeros((ndim+1,ndim+1), N.Float)
    ptransform[0:ndim,0:ndim] = permutation_matrix(order=order)
    ptransform[ndim,ndim] = 1.

    return ptransform

def _translation_transform(x, ndim):
    """
    Create an affine transformation matrix representing translation by x.
    """
    _transform = N.identity(ndim+1)
    _transform[0:ndim,ndim] = _transform[0:ndim,ndim] + x 
    return _transform

def tovoxel(real, mapping):
    """
    Given a mapping and a real coordinate, where mapping.input_coords are assumed to be voxels, return the closest voxel for real. Will choke if mapping is not invertible.
    """
    _shape = real.shape
    real.shape = (_shape[0], product(_shape[1:]))
    voxel = N.around(mapping.map(real, inverse=True))
    real.shape = _shape
    voxel.shape = _shape
    return N.array(voxel)

def matlab2python(mapping):
    """
    Take that maps matlab voxels to (matlab-ordered) world coordinates and make it python-oriented. This means that if mapping(v_x,v_y,v_z)=(w_x,w_y,w_z) then the return will send (v_z,v_y,v_x) to (w_z,w_y,w_x).
    """

    ndim = mapping.input_coords.ndim
    t1 = N.zeros((ndim+1,)*2, N.Float)
    t1[0:ndim,0:ndim] = permutation_matrix(range(ndim)[::-1])
    t1[ndim, ndim] = 1.0

    t2 = 1. * t1
    t1[0:ndim,ndim] = 1.0

    n = mapping.ndim
    d1 = [mapping.input_coords.axes[n-1-i] for i in range(n)]
    in1 = CoordinateSystem(mapping.input_coords.name, d1)
    w1 = Affine(in1, mapping.input_coords, t1)
    
    d2 = [mapping.output_coords.axes[n-1-i] for i in range(n)]
    out2 = CoordinateSystem(mapping.output_coords.name, d2)
    w2 = Affine(mapping.output_coords, out2, t2)

    w = (w2 * mapping) * w1
    return w

fortran2C = matlab2python

def python2matlab(mapping):
    """
    Inverse of matlab2python -- see this function for help.
    """

    ndim = mapping.input_coords.ndim
    t1 = N.zeros((ndim+1,)*2, N.Float)
    t1[0:ndim,0:ndim] = permutation_matrix(range(ndim)[::-1])
    t1[ndim, ndim] = 1.0

    t2 = 1. * t1
    t1[0:ndim,ndim] = -1.0

    n = mapping.ndim
    d1 = [mapping.input_coords.axes[n-1-i] for i in range(n)]
    in1 = CoordinateSystem(mapping.input_coords.name, d1)
    w1 = Affine(in1, mapping.input_coords, t1)
    
    d2 = [mapping.output_coords.axes[n-1-i] for i in range(n)]
    out2 = CoordinateSystem(mapping.output_coords.name, d2)
    w2 = Affine(mapping.output_coords, out2, t2)

    w = (w2 * mapping) * w1

    return w

C2fortran = python2matlab

MNI_mapping = Affine(MNI_voxel, MNI_world, MNI_world.transform())
MNI_mapping([36,63,45])
