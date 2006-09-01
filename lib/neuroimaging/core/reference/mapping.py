import csv, urllib
from struct import unpack

import numpy as N
from numpy.linalg import inv


from neuroimaging import hasattrs
from neuroimaging.core.reference.axis import VoxelAxis, space
from neuroimaging.core.reference.coordinate_system import \
  VoxelCoordinateSystem, DiagonalCoordinateSystem

def _2matvec(transform):
    """ Split a tranform into it's matrix and vector components. """
    ndim = transform.shape[0] - 1
    matrix = transform[0:ndim, 0:ndim]
    vector = transform[0:ndim, ndim]
    return matrix, vector

def _2transform(matrix, vector):
    """ Combine a matrix and vector into a transform """
    nin = matrix.shape[1]
    t = N.zeros((nin+1,)*2)
    t[0:nin, 0:nin] = matrix
    t[nin,   nin] = 1.
    t[0:nin, nin] = vector
    return t
    

def matfromfile(infile, delimiter="\t"):
    """ Read in an affine transformation matrix from a csv file."""
    if type(infile)==type(""): infile = open(infile)
    reader = csv.reader(infile, delimiter=delimiter)
    return N.array([map(float, row) for row in reader])

def frombin(tstr):
    """
    This is broken -- anyone with mat file experience?
    >>> import urllib
    >>> from neuroimaging.core.reference.mapping import frombin
    >>> mat = urllib.urlopen('http://kff.stanford.edu/BrainSTAT/fiac3_fonc1_0089.mat')
    >>> tstr = mat.read()
    >>> print frombin(tstr)
    [[  2.99893500e+00  -3.14532000e-03  -1.06594400e-01  -9.61109780e+01]
    [ -1.37396100e-02  -2.97339600e+00  -5.31224000e-01   1.20082725e+02]
    [  7.88193000e-02  -3.98643000e-01   3.96313600e+00  -3.32398676e+01]
    [  0.00000000e+00   0.00000000e+00   0.00000000e+00   1.00000000e+00]]
    
    """

    T = N.array(unpack('<16d', tstr[-128:]))
    T.shape = (4,4)
    T = N.transpose(T)
    return T

def matfromstr(tstr, ndim=3, delimiter=None):
    "Read a (ndim+1)x(ndim+1) transform matrix from a string."
    if tstr[0:24] == "mat file created by perl":
        return frombin(tstr) 
    else:
        transform = N.array(map(float, tstr.split(delimiter)))
        transform.shape = (ndim+1,)*2
        return transform


def xfmfromstr(tstr, ndim=3):
    "Read a (ndim+1)x(ndim+1) transform matrix from a string."
    tstr = tstr.split('\n')
    more = True
    data = []
    outdata = []
    for i in range(len(tstr)):

        if tstr[i].find('/matrix') >= 0:
            for j in range((ndim+1)**2):
                data.append(float(tstr[i+j+1]))

        if tstr[i].find('/outputusermatrix') >= 0:
            for j in range((ndim+1)**2):
                outdata.append(float(tstr[i+j+1]))

    data = N.array(data)
    data.shape = (ndim+1,)*2
    outdata = N.array(outdata)
    outdata.shape = (ndim+1,)*2
    return data, outdata


def fromurl(turl, ndim=3):
    """
    Read a (ndim+1)x(ndim+1) transform matrix from a URL -- tries to autodetect
    '.mat' and '.xfm'.

    >>> from numpy import testing
    >>> from neuroimaging.core.reference.mapping import fromurl
    >>> x = fromurl('http://kff.stanford.edu/nipy/testdata/fiac3_fonc1.txt')
    >>> y = fromurl('http://kff.stanford.edu/nipy/testdata/fiac3_fonc1_0089.mat')
    >>> print testing.assert_almost_equal(x, y)
    True
    """
    urlpipe = urllib.urlopen(turl)
    data = urlpipe.read()
    if turl[-3:] in ['mat', 'txt']: return matfromstr(data, ndim=ndim)
    elif turl[-3:] == 'xfm': return xfmfromstr(data, ndim=ndim)


def isdiagonal(matrix, tol=1.0e-7):
    ndim = matrix.shape[0]
    D = N.diag(N.diagonal(matrix))
    dmatrix = matrix - D
    dmatrix.shape = (ndim,)*2
    dnorm = N.add.reduce(dmatrix**2)
    fmatrix = 1. * matrix
    fmatrix.shape = dmatrix.shape
    norm = N.add.reduce(fmatrix**2)
    return N.add.reduce(dnorm / norm) < tol



class Mapping (object):
    """
    A generic mapping class that allows composition, inverses, etc. A mapping
    needs only input and output coordinates and a transform between the two,
    and an optional inverse.
    """

    def __init__(self, input_coords, output_coords, map, inverse=None, name="mapping"):
        self.input_coords = input_coords
        self.output_coords = output_coords
        self.map = map
        self._inverse = inverse
        self.name = name


    def __call__(self, x):
        return self.map(x)


    def __str__(self):
        return '%s:input=%s\n'%(self.name, self.input_coords) +\
        '%s:output=%s\n'%(self.name, self.output_coords) +\
        '%s:map=%s\n'%(self.name, self.map) +\
        '%s:inverse=%s\n'%(self.name, self._inverse)


    def __ne__(self, other): return not self.__eq__(other)
    def __eq__(self, other):
        if not hasattrs(other, "input_coords", "output_coords", "map"):
            return False
        return (self.input_coords, self.output_coords, self.map) == \
               (other.input_coords, other.output_coords, other.map)


    def __mul__(self, other):
        "If this method is not over-written we get complaints about sequences."
        return other.__rmul__(self)
    

    def __rmul__(self, other):
        def map(coords): 
            return other(self(coords))
        if self.isinvertible() and other.isinvertible():
            def inverse(coords): 
                return self.inverse()(other.inverse()(coords))
        else: 
            inverse = None
        return Mapping(self.input_coords, other.output_coords, map, inverse=inverse)

    def isinvertible(self):
        return self._inverse is not None

    def inverse(self):
        if self.isinvertible():
            return Mapping(self.output_coords, self.input_coords,
                self._inverse, self)
        else: 
            raise AttributeError("non-invertible mapping")


    def ndim(self):
        return self.input_coords.ndim()

    def tovoxel(self, real):
        """
        Given a real coordinate, where self.input_coords are assumed to be
        voxels, return the closest voxel for real. Will choke if mapping is
        not invertible.
        """
        shape = real.shape
        real.shape = (shape[0], N.product(shape[1:]))
        voxel = N.around(self.inverse()(real))
        voxel.shape = shape
        return N.array(voxel)


    def matlab2python(self):
        """
        Take that maps matlab voxels to (matlab-ordered) world coordinates and
        make it python-oriented. This means that if
        mapping(v_x,v_y,v_z)=(w_x,w_y,w_z), then the return will send
        (v_z-1,v_y-1,v_x-1) to (w_z,w_y,w_x).

        >>> from neuroimaging.core.image import Image
        >>> zimage = Image('http://nifti.nimh.nih.gov/nifti-1/data/zstat1.nii.gz')
        >>> mapping = zimage.grid.mapping
        >>> mapping([1,2,3])
        array([ 2.,  3., -4.])

        >>> matlab = mapping.python2matlab()
        >>> matlab([4,3,2])
        array([-4.,  3.,  2.])
        >>>

        """
        return self._f(1.0)

    def python2matlab(self):
        "Inverse of matlab2python -- see this function for help."
        return self._f(-1.0)

    def _f(self, x):
        """ helper function for matlab2python and python2matlab """
        ndim = self.ndim()
        mat = permutation_matrix(range(ndim)[::-1])
        t1 = _2transform(mat, x)
        t2 = _2transform(mat, N.zeros(ndim))
        w1 = Affine(self.input_coords.reverse(), self.input_coords, t1)
        w2 = Affine(self.output_coords, self.output_coords.reverse(), t2)
        return (w2 * self) * w1


class Affine(Mapping):
    "A class representing an affine transformation in n axes."

    @staticmethod
    def frommatrix(matrix, names=space, input='voxel', output='world'):
        """
        Return an Affine instance with named axes and input and output
        coordinate systems.
        """
        axes = [VoxelAxis(name) for name in names]
        return Affine(
          VoxelCoordinateSystem(input, axes), 
          DiagonalCoordinateSystem(output, axes),
          matrix)


    @staticmethod
    def fromfile(infile, names=space, input='voxel', output='world', delimiter='\t'):
        """
        Read in an affine transformation matrix and return an instance of Affine
        with named axes and input and output coordinate systems.  For now, the
        format is assumed to be a tab-delimited file.  Other formats should be added.
        """
        t = matfromfile(infile, delimiter=delimiter)
        return Affine.frommatrix(t, names=names, input=input, output=output)


    @staticmethod
    def identity(ndim=3, names=space, input='voxel', output='world'):
        "Return an identity affine transformation."
        return Affine.frommatrix(
          N.identity(ndim+1), names=names, input=input, output=output)


    def __init__(self, input_coords, output_coords, transform, name="affine"):
        self.transform = transform
        self._fmatrix, self._fvector = _2matvec(transform)
        Mapping.__init__(self, input_coords, output_coords, self.map, name=name)


    def __eq__(self, other):
        if not hasattr(other, "transform"): return False
        return tuple(self.transform.flat) == tuple(other.transform.flat)


    def __rmul__(self, other):
        if isinstance(other, Affine):
            try: return Affine(
              self.input_coords, other.output_coords,
                N.dot(other.transform, self.transform))
            except:
                fmatrix = N.dot(other._fmatrix, self._fmatrix)
                fvector = N.dot(other._fmatrix, self._fvector) + other._fvector
                return DegenerateAffine(
                  self.input_coords, other.output_coords, fmatrix, fvector)
        else: return Mapping.__rmul__(self, other)


    def __str__(self):
        return "%s:input=%s\n%s:output=%s\n%s:fmatrix=%s\n%s:fvector=%s" % \
          (self.name, self.input_coords.name, self.name,
           self.output_coords.name, self.name, `self._fmatrix`, self.name,
           `self._fvector`)
 

    def map(self, coords):
        value = N.dot(self._fmatrix, coords) 
        value = value + N.multiply.outer(self._fvector, N.ones(value.shape[1:]))
        return value
    

    def inverse(self):
        return Affine(self.output_coords, self.input_coords,
                      inv(self.transform))
    
    def isinvertible(self):
        return True # is this true?

    def isdiagonal(self):
        return isdiagonal(self._fmatrix)
 

    def tofile(self, filename):
        matfile = open(filename, 'w')
        writer = csv.writer(matfile, delimiter='\t')
        for row in self.transform: writer.writerow(row)
        matfile.close()
  


class DegenerateAffine(Affine):
    """
    A subclass of affine with no inverse, i.e. where the map is non-invertible.
    """

    def __init__(self, input_coords, output_coords, fmatrix, fvector,
                 name='transform'):

        def map(coords):
            value = N.dot(fmatrix, coords) 
            value = value + N.multiply.outer(fvector, N.ones(value.shape[1:]))
            return value

        try:
            inv(fmatrix) # if not invertable we want to raise here
            t = _2transform(fmatrix, fvector)
            Affine.__init__(self, input_coords, output_coords, t, name=name)
        except:
            Mapping.__init__(self, input_coords, output_coords, map, name=name)


def permutation_matrix(order=range(3)[2::-1]):
    """
    Create an NxN permutation matrix from a sequence, containing the values 0,...,N-1.
    """
    n = len(order)
    matrix = N.zeros((n, n))
    if set(order) != set(range(n)):
        raise ValueError(
          'order should be a sequence of integers with values, 0 ... len(order)-1.')
    for i in range(n): 
        matrix[i,order[i]] = 1
    return matrix


def permutation_transform(order=range(3)[2::-1]):
    """
    Create an (N+1)x(N+1) permutation transformation matrix from a sequence,
    containing the values 0,...,N-1.
    """
    matrix = permutation_matrix(order=order)
    vector = N.zeros(len(order))
    return _2transform(matrix, vector)


def translation_transform(x, ndim):
    """
    Create an affine transformation matrix representing translation by x.
    """
    return _2transform(N.identity(ndim), x)
