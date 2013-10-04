# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
from scipy.stats import norm
TINY = 1e-16


def z_score(pvalue):
    """ Return the z-score corresponding to a given p-value.
    """
    pvalue = np.minimum(np.maximum(pvalue, 1.e-300), 1. - TINY)
    z = norm.isf(pvalue)
    return z


def multiple_fast_inv(a):
    """Compute the inverse of a set of arrays.

    Parameters
    ----------
    a: array_like of shape (n_samples, n_dim, n_dim)
        Set of square matrices to be inverted. A is changed in place.

    Returns
    -------
    a: ndarray
       yielding the inverse of the inputs

    Raises
    ------
    LinAlgError :
        If `a` is singular.
    ValueError :
        If `a` is not square, or not 2-dimensional.

    Notes
    -----
    This function is borrowed from scipy.linalg.inv, 
    but with some customizations for speed-up.
    """
    if a.shape[1] != a.shape[2]:
        raise ValueError('a must have shape(n_samples, n_dim, n_dim)')
    from scipy.linalg import calc_lwork
    from scipy.linalg.lapack import  get_lapack_funcs
    a1, n = a[0], a.shape[0]
    getrf, getri = get_lapack_funcs(('getrf', 'getri'), (a1,))
    for i in range(n):
        if (getrf.module_name[:7] == 'clapack'
            and getri.module_name[:7] != 'clapack'):
            # ATLAS 3.2.1 has getrf but not getri.
            lu, piv, info = getrf(np.transpose(a[i]), rowmajor=0,
                                  overwrite_a=True)
            a[i] = np.transpose(lu)
        else:
            a[i], piv, info = getrf(a[i], overwrite_a=True)
        if info == 0:
            if getri.module_name[:7] == 'flapack':
                lwork = calc_lwork.getri(getri.prefix, a1.shape[0])
                lwork = lwork[1]
                # XXX: the following line fixes curious SEGFAULT when
                # benchmarking 500x500 matrix inverse. This seems to
                # be a bug in LAPACK ?getri routine because if lwork is
                # minimal (when using lwork[0] instead of lwork[1]) then
                # all tests pass. Further investigation is required if
                # more such SEGFAULTs occur.
                lwork = int(1.01 * lwork)
                a[i], _ = getri(a[i], piv, lwork=lwork, overwrite_lu=1)
            else:  # clapack
                a[i], _ = getri(a[i], piv, overwrite_lu=1)
        else:
            raise ValueError('Matrix LU decomposition failed')
    return a


def multiple_mahalanobis(effect, covariance):
    """Returns the squared Mahalanobis distance for a given set of samples
    
    Parameters
    ----------
    effect: array of shape (n_features, n_samples),
            Each column represents a vector to be evaluated
    covariance: array of shape (n_features, n_features, n_samples),
                Corresponding covariance models stacked along the last axis

    Returns
    -------
    sqd: array of shape (n_samples,)
         the squared distances (one per sample)
    """ 
    # check size
    if effect.ndim == 1:
        effect = effect[:, np.newaxis]
    if covariance.ndim == 2:
        covariance = covariance[:, :, np.newaxis]
    if effect.shape[0] != covariance.shape[0]:
        raise ValueError('Inconsistant shape for effect and covariance')
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError('Inconsistant shape for covariance')

    # transpose and make contuguous for the sake of speed
    Xt, Kt = np.ascontiguousarray(effect.T), np.ascontiguousarray(covariance.T)

    # compute the inverse of the covariances
    Kt = multiple_fast_inv(Kt)
    
    # derive the squared Mahalanobis distances
    sqd = np.sum(np.sum(Xt[:, :, np.newaxis] * Xt[:, np.newaxis] * Kt, 1), 1)
    return sqd


# Taken from python doc site, exists in python2.6
def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)


def complex(maximal=[(0, 3, 2, 7),
                     (0, 6, 2, 7),
                     (0, 7, 5, 4),
                     (0, 7, 5, 1),
                     (0, 7, 4, 6),
                     (0, 3, 1, 7)]):
    """ Faces from simplices
    
    Take a list of maximal simplices (by default a triangulation of a
    cube into 6 tetrahedra) and computes all faces

    Parameters
    ----------
    maximal : sequence of sequences, optional
       Default is triangulation of cube into tetrahedra

    Returns
    -------
    faces : dict
    """
    faces = {}

    l = [len(list(x)) for x in maximal]
    for i in range(np.max(l)):
        faces[i+1] = set([])

    for simplex in maximal:
        simplex = list(simplex)
        simplex.sort()
        for k in range(1,len(simplex)+1):
            for v in combinations(simplex, k):
                if len(v) == 1:
                    v = v[0]
                faces[k].add(v)
    return faces


def cube_with_strides_center(center=[0,0,0],
                             strides=[4, 2, 1]):
    """ Cube in an array of voxels with a given center and strides.

    This triangulates a cube with vertices [center[i] + 1].
    
    The dimension of the cube is determined by len(center)
    which should agree with len(center). 

    The allowable dimensions are [1,2,3].

    Parameters
    ----------
    center : (d,) sequence of int, optional
       Default is [0, 0, 0]
    strides : (d,) sequence of int, optional
       Default is [4, 2, 1].  These are the strides given by
       ``np.ones((2,2,2), np.bool).strides``
       
    Returns
    -------
    complex : dict
       A dictionary with integer keys representing a simplicial
       complex. The vertices of the simplicial complex are the indices
       of the corners of the cube in a 'flattened' array with specified
       strides.
    """
    d = len(center)
    if not 0 < d <= 3:
        raise ValueError('dimensionality must be 0 < d <= 3')
    if len(strides) != d:
        raise ValueError('center and strides must have the same length')
    if d == 3:
        maximal = [(0, 3, 2, 7),
                   (0, 6, 2, 7),
                   (0, 7, 5, 4),
                   (0, 7, 5, 1),
                   (0, 7, 4, 6),
                   (0, 3, 1, 7)]
        vertices = []
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    vertices.append((center[0]+i)*strides[0] + 
                                    (center[1]+j)*strides[1] +
                                    (center[2]+k)*strides[2])
    elif d == 2:
        maximal = [(0,1,3), (0,2,3)]
        vertices = []
        for j in range(2):
            for i in range(2):
                    vertices.append((center[0]+i)*strides[0] + 
                                    (center[1]+j)*strides[1])
    elif d == 1:
        maximal = [(0,1)]
        vertices = [center[0],center[0]+strides[0]]

    mm = []
    for m in maximal:
        nm = [vertices[j] for j in m]
        mm.append(nm)
    maximal = [tuple([vertices[j] for j in m]) for m in maximal]
    return complex(maximal)


def join_complexes(*complexes):
    """ Join a sequence of simplicial complexes.
    
    Returns the union of all the particular faces.
    """
    faces = {}

    nmax = np.array([len(c) for c in complexes]).max()
    for i in range(nmax):
        faces[i+1] = set([])
    for c in complexes:
        for i in range(nmax):
            if c.has_key(i+1):
                faces[i+1] = faces[i+1].union(c[i+1])
    return faces

def decompose3d(shape, dim=4):
    """
    Return all (dim-1)-dimensional simplices in a triangulation
    of a cube of a given shape. The vertices in the triangulation
    are indices in a 'flattened' array of the specified shape.
    """

    # First do the interior contributions.
    # We first figure out which vertices, edges, triangles, tetrahedra
    # are uniquely associated with an interior voxel

    unique = {}
    strides = np.empty(shape, np.bool).strides
    union = join_complexes(*[cube_with_strides_center((0,0,-1), strides),
                             cube_with_strides_center((0,-1,0), strides),
                             cube_with_strides_center((0,-1,-1), strides),
                             cube_with_strides_center((-1,0,0), strides),
                             cube_with_strides_center((-1,0,-1), strides),
                             cube_with_strides_center((-1,-1,0), strides),
                             cube_with_strides_center((-1,-1,-1), strides)])

    c = cube_with_strides_center((0,0,0), strides)
    for i in range(4):
        unique[i+1] = c[i+1].difference(union[i+1])

    if unique.has_key(dim) and dim > 1:
        d = unique[dim]

        for i in range(shape[0]-1):
            for j in range(shape[1]-1):
                for k in range(shape[2]-1):
                    index = i*strides[0]+j*strides[1]+k*strides[2]
                    for l in d:
                        yield [index+ii for ii in l]

    # There are now contributions from three two-dimensional faces

    for _strides, _shape in zip([(strides[0], strides[1]), 
                                 (strides[0], strides[2]),
                                 (strides[1], strides[2])],
                                [(shape[0], shape[1]),
                                 (shape[0], shape[2]),
                                 (shape[1], shape[2])]):
        
        unique = {}
        union = join_complexes(*[cube_with_strides_center((0,-1), _strides),
                                 cube_with_strides_center((-1,0), _strides),
                                 cube_with_strides_center((-1,-1), _strides)])

        c = cube_with_strides_center((0,0), _strides)
        for i in range(3):
            unique[i+1] = c[i+1].difference(union[i+1])
        
        if unique.has_key(dim) and dim > 1:
            d = unique[dim]

            for i in range(_shape[0]-1):
                for j in range(_shape[1]-1):
                        index = i*_strides[0]+j*_strides[1]
                        for l in d:
                            yield [index+ii for ii in l]

    # Finally the one-dimensional faces

    for _stride, _shape in zip(strides, shape):
        
        unique = {}
        union = cube_with_strides_center((-1,), [_stride])
        c = cube_with_strides_center((0,), [_stride])
        for i in range(2):
            unique[i+1] = c[i+1].difference(union[i+1])

        if unique.has_key(dim) and dim > 1:
            d = unique[dim]

            for i in range(_shape-1):
                index = i*_stride
                for l in d:
                    yield [index+ii for ii in l]

    if dim == 1:
        for i in range(np.product(shape)):
            yield i


def decompose2d(shape, dim=3):
    """
    Return all (dim-1)-dimensional simplices in a triangulation
    of a square of a given shape. The vertices in the triangulation
    are indices in a 'flattened' array of the specified shape.
    """
    # First do the interior contributions.
    # We first figure out which vertices, edges, triangles
    # are uniquely associated with an interior pixel

    unique = {}
    strides = np.empty(shape, np.bool).strides
    union = join_complexes(*[cube_with_strides_center((0,-1), strides),
                             cube_with_strides_center((-1,0), strides),
                             cube_with_strides_center((-1,-1), strides)])
    c = cube_with_strides_center((0,0), strides)
    for i in range(3):
        unique[i+1] = c[i+1].difference(union[i+1])

    if unique.has_key(dim) and dim > 1:
        d = unique[dim]

        for i in range(shape[0]-1):
            for j in range(shape[1]-1):
                    index = i*strides[0]+j*strides[1]
                    for l in d:
                        yield [index+ii for ii in l]

    # Now, the one-dimensional faces

    for _stride, _shape in zip(strides, shape):
        
        unique = {}
        union = cube_with_strides_center((-1,), [_stride])
        c = cube_with_strides_center((0,), [_stride])
        for i in range(2):
            unique[i+1] = c[i+1].difference(union[i+1])

        if unique.has_key(dim) and dim > 1:
            d = unique[dim]

            for i in range(_shape-1):
                index = i*_stride
                for l in d:
                    yield [index+ii for ii in l]

    if dim == 1:
        for i in range(np.product(shape)):
            yield i


def test_EC3(shape):

    ts = 0
    fs = 0
    es = 0
    vs = 0
    ec = 0

    for t in decompose3d(shape, dim=4):
        ec -= 1; ts += 1
    for f in decompose3d(shape, dim=3):
        ec += 1; fs += 1
    for e in decompose3d(shape, dim=2):
        ec -= 1; es += 1
    for v in decompose3d(shape, dim=1):
        ec += 1; vs += 1
    return ts, fs, es, vs, ec

# Tell nose testing framework not to run this as a test
test_EC3.__test__ = False


def test_EC2(shape):

    fs = 0
    es = 0
    vs = 0
    ec = 0

    for f in decompose2d(shape, dim=3):
        ec += 1; fs += 1
    for e in decompose2d(shape, dim=2):
        ec -= 1; es += 1
    for v in decompose2d(shape, dim=1):
        ec += 1; vs += 1
    return fs, es, vs, ec

# Tell nose testing framework not to run this as a test
test_EC2.__test__ = False


