"""
Mappings define a transformation between sets of coordinate systems.

These mappings can be used to transform between voxel space and real space,
for example.
"""

__docformat__ = 'restructuredtext'

import csv
import urllib
from struct import unpack

import numpy as N
from numpy.linalg import inv

def _2matvec(transform):
    """ Split a tranform into it's matrix and vector components. """
    ndim = transform.shape[0] - 1
    matrix = transform[0:ndim, 0:ndim]
    vector = transform[0:ndim, ndim]
    return matrix, vector

def _2transform(matrix, vector):
    """ Combine a matrix and vector into a transform. """
    nin = matrix.shape[1]
    t = N.zeros((nin+1,)*2)
    t[0:nin, 0:nin] = matrix
    t[nin,   nin] = 1.
    t[0:nin, nin] = vector
    return t


def permutation_matrix(order=range(3)[2::-1]):
    """
    Create an NxN permutation matrix from a sequence, containing the values
    0, ..., N-1.
    """
    n = len(order)
    matrix = N.zeros((n, n))
    if set(order) != set(range(n)):
        raise ValueError('order should be a sequence of integers with' \
                         'values, 0 ... len(order)-1.')
    for i in range(n): 
        matrix[i, order[i]] = 1
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
    

def matfromfile(infile, delimiter="\t"):
    """ Read in an affine transformation matrix from a csv file."""
    if isinstance(infile, str):
        infile = open(infile)
    reader = csv.reader(infile, delimiter=delimiter)
    return N.array(list(reader)).astype(float)

def frombin(tstr):
    """
    This is broken -- anyone with mat file experience?

    >>> import urllib
    >>> from neuroimaging.core.reference.mapping import frombin
    >>> mat = urllib.urlopen('http://kff.stanford.edu/nipy/testdata/fiac3_fonc1_0089.mat')
    >>> tstr = mat.read()
    >>> print frombin(tstr)
    [[  2.99893500e+00  -3.14532000e-03  -1.06594400e-01  -9.61109780e+01]
     [ -1.37396100e-02  -2.97339600e+00  -5.31224000e-01   1.20082725e+02]
     [  7.88193000e-02  -3.98643000e-01   3.96313600e+00  -3.32398676e+01]
     [  0.00000000e+00   0.00000000e+00   0.00000000e+00   1.00000000e+00]]
    
    """

    T = N.array(unpack('<16d', tstr[-128:]))
    T.shape = (4, 4)
    return T.T

def matfromstr(tstr, ndim=3, delimiter=None):
    """Read a (ndim+1)x(ndim+1) transform matrix from a string."""
    if tstr[0:24] == "mat file created by perl":
        return frombin(tstr) 
    else:
        transform = N.array(tstr.split(delimiter)).astype(float)
        transform.shape = (ndim+1,)*2
        return transform


def xfmfromstr(tstr, ndim=3):
    """Read a (ndim+1)x(ndim+1) transform matrix from a string.

    The format being read is that used by the FLS group, for example
    http://kff.stanford.edu/FIAC/fiac0/fonc1/fsl/example_func2highres.xfm
    """
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
    >>> testing.assert_almost_equal(x, y, decimal=5)

    """
    urlpipe = urllib.urlopen(turl)
    data = urlpipe.read()
    if turl[-3:] in ['mat', 'txt']:
        return matfromstr(data, ndim=ndim)
    elif turl[-3:] == 'xfm':
        return xfmfromstr(data, ndim=ndim)


def isdiagonal(matrix):
    """
    Test if the given matrix is diagonal.

    :Parameters:
        `matrix` : numpy.ndarray
            The matrix to check.
    :Returns: ``bool``

    :Precondition: matrix must be square
    """

    ndim = matrix.shape[0]
    mask = ~N.identity(ndim, dtype=bool)
    masked = N.ma.array(matrix, mask=mask, fill_value=0.).filled()
    return N.all(masked == matrix)


class Mapping(object):
    """
    A generic mapping class that allows composition, inverses, etc. A mapping
    needs only input and output coordinates and a transform between the two,
    and an optional inverse.
    """

    def __init__(self, map, inverse=None, name="mapping", ndim=3):
        """
        :Parameters:
            `map` : callable
                A function which takes a coordinate vector and returns a new coordinate vector
            `inverse` : callable
                An inverse function to ``map``
            `name` : string
                The name of the mapping
            `ndim` : int
                The number of dimensions of the mapping.
        """
        self._map = map
        self._inverse = inverse
        self.name = name
        self._ndim = ndim
        
        
    def __call__(self, x):
        """ Apply this mapping to the given coordinates. """
        return self._map(x)


    def __str__(self):
        return '%s:map=%s\n' % (self.name, self._map) + \
        '%s:inverse=%s\n' % (self.name, self._inverse)


    def __ne__(self, other): 
        """
        :SeeAlso:
         - `Mapping.__eq__`
        """
        return not self.__eq__(other)
        
    def __eq__(self, other):
        """
        We can't say whether two map functions are the same so we just
        raise an exception if people try to compare mappings.

        :Raises NotImplementedError:
        """
        raise NotImplementedError

    def __mul__(self, other):
        """If this method is not over-written we get complaints about sequences.

        :Returns: `Mapping`
        """
        return other.__rmul__(self)
    

    def __rmul__(self, other):
        """ mapping composition

        :Parameters:
            `other` : `Mapping`
                The mapping to compose with.
        :Returns: `Mapping`
        """
        def map(coords): 
            return other(self(coords))
        if self.isinvertible() and other.isinvertible():
            def inverse(coords): 
                return self.inverse()(other.inverse()(coords))
        else: 
            inverse = None
        return Mapping(map, inverse=inverse)

    def ndim(self):
        """ The number of input dimensions

        :Returns: ``int``
        """
        return self._ndim

    def isinvertible(self):
        """
        Does this mapping have an inverse?

        :Returns: ``bool``
        """
        return self._inverse is not None

    def inverse(self):
        """
        Create a new Mapping instance which is the inverse of self.

        :Returns:
            `Mapping`
        """
        if self.isinvertible():
            return Mapping(self._inverse, self)
        else: 
            raise AttributeError("non-invertible mapping")

    def tovoxel(self, real):
        """
        Given a real coordinate, where self.input_coords are assumed to be
        voxels, return the closest voxel for real. Will choke if mapping is
        not invertible.

        :Raises N.linalg.LinAlgError: is mapping is not invertible.
        """
        shape = real.shape
        if len(shape) > 1:
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

        :Returns:
            `Mapping`

        Examples
        --------

        >>> from neuroimaging.core.image.image import Image
        >>> zimage = Image('http://nifti.nimh.nih.gov/nifti-1/data/zstat1.nii.gz')
        >>> mapping = zimage.grid.mapping
        >>> mapping([1,2,3])
        array([ 12.,  12., -16.])

        >>> matlab = mapping.python2matlab()
        >>> matlab([4,3,2])
        array([-16.,  12.,  12.])
        >>>

        """
        return self._f(1.0)

    def python2matlab(self):
        """ Inverse of `matlab2python` -- see this function for help.

        :Returns:
            `Mapping`
        """
        return self._f(-1.0)

    def _f(self, x):
        """ helper function for `matlab2python` and `python2matlab` """
        ndim = self.ndim()
        mat = permutation_matrix(range(ndim)[::-1])
        t1 = _2transform(mat, x)
        t2 = _2transform(mat, N.zeros(ndim))
        w1 = Affine(t1)
        w2 = Affine(t2)
        return (w2 * self) * w1


class Affine(Mapping):
    """
    A class representing an affine transformation in n axes.
    
    This class adds a transform member, which is a matrix representing
    the affine transformation. This matrix is used to perform mappings,
    rather than having an explicit mapping function. 
    """

    @staticmethod
    def fromfile(infile, delimiter='\t'):
        """
        Read in an affine transformation matrix and return an instance of
        `Affine` with named axes and input and output coordinate systems.  For
        now, the format is assumed to be a tab-delimited file.  Other formats
        should be added.

        :Returns:
            `Affine`
        """
        t = matfromfile(infile, delimiter=delimiter)
        return Affine(t)


    @staticmethod
    def identity(ndim=3):
        """ Return an identity affine transformation.

        :Returns:
            `Affine`
        """
        return Affine(N.identity(ndim+1))


    def __init__(self, transform, name="affine"):
        """
        :Parameters:
            `transform` : numpy.ndarray
                A transformation matrix
            `name` : string
                The name of the mapping
        """
        self.transform = transform
        ndim = transform.shape[0] - 1
        self._fmatrix, self._fvector = _2matvec(transform)
        Mapping.__init__(self, None, name=name, ndim=ndim)

    def __call__(self, coords):
        """ Apply this mapping to the given coordinates. 
        
        :Parameters:
            `coords` : numpy.ndarray
                A coordinate vector
        
        :Returns: ``numpy.ndarray``
        """
        value = N.dot(self._fmatrix, coords) 
        value += N.multiply.outer(self._fvector, N.ones(value.shape[1:]))
        return value

    def __eq__(self, other):
        """
        Equality is defined as equality of both name and transform matrix.

        :Parameters:
            `other` : `Affine`
                The mapping to be compared to.
                
        :Returns: ``bool``
        """
        if not hasattr(other, "transform"): 
            return False
        return N.all(N.asarray(self.transform) == N.asarray(other.transform)) \
               and self.name == other.name


    def __rmul__(self, other):
        """
        :Parameters:
            `other` : `Mapping` or `Affine`
                The mapping to be multiplied by.
                
        :Returns: `Mapping` or `Affine`
        """
        if isinstance(other, Affine):
            return Affine(N.dot(other.transform, self.transform))            
        else: 
            return Mapping.__rmul__(self, other)


    def __str__(self):
        """
        :Returns: ``string``
        """
        return "%s:fmatrix=%s\n%s:fvector=%s" % \
          (self.name, `self._fmatrix`, self.name,`self._fvector`)
 

    def isinvertible(self):
        """
        Does this mapping have an inverse?

        :Returns: ``bool``
        """
        try:
            inv(self.transform)
            return True
        except:
            return False

    def inverse(self):
        """
        Create a new `Affine` instance which is the inverse of self.

        :Returns: `Affine`
        """
        return Affine(inv(self.transform))

    
    def isdiagonal(self):
        """
        Is the transform matrix diagonal?

        :Returns: ``bool``
        """
        return isdiagonal(self._fmatrix)

 
    def tofile(self, filename):
        """
        Write the transform matrix to a file.

        :Parameters:
            `filename` : string
                The filename to write to

        :Returns: ``None``
        """
        matfile = open(filename, 'w')
        writer = csv.writer(matfile, delimiter='\t')
        for row in self.transform: 
            writer.writerow(row)
        matfile.close()
  
