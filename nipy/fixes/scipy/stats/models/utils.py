''' General matrix and other utilities for statistics '''

import numpy as np
import numpy.linalg as npl
import scipy.interpolate

__docformat__ = 'restructuredtext'


def pos_recipr(X):
    """ Return element-wise reciprocal of array, setting `X`>=0 to 0
    
    Return the reciprocal of an array, setting all entries less than or
    equal to 0 to 0. Therefore, it presumes that X should be positive in
    general.

    Parameters
    ----------
    X : array-like

    Returns
    -------
    rX : array
       array of same shape as `X`, dtype np.float, with values set to
       1/X where X > 0, 0 otherwise
    """
    X = np.asarray(X)
    rX = np.zeros(X.shape)
    gt0 = X > 0
    rX[gt0] = 1. / X[gt0]
    return rX


def recipr0(X):
    """ Return element-wise reciprocal of array, `X`==0 -> 0
    
    Return the reciprocal of an array, setting all entries equal to 0
    as 0. It does not assume that X should be positive in
    general.

    Parameters
    ----------
    X : array-like

    Returns
    -------
    rX : array
    """
    test = np.equal(np.asarray(X), 0)
    return np.where(test, 0, 1. / X)


def mad(a, c=0.6745, axis=0):
    """
    Median Absolute Deviation:

    median(abs(a - median(a))) / c

    """

    _shape = a.shape
    a.shape = np.product(a.shape,axis=0)
    m = np.median(np.fabs(a - np.median(a))) / c
    a.shape = _shape
    return m


def matrix_rank(M, tol=None):
    ''' Return rank of matrix using SVD method

    Rank of the array is the number of SVD singular values of the
    array that are greater than `tol`.
    
    Parameters
    ----------
    M : array-like
        array of <=2 dimensions
    tol : {None, float}
         threshold below which SVD values are considered zero. If `tol`
         is None, and `S` is an array with singular values for `M`, and
         `eps` is the epsilon value for datatype of `S`, then `tol` set
         to ``S.max() * eps``.

    Examples
    --------
    >>> matrix_rank(np.eye(4)) # Full rank matrix
    4
    >>> I=np.eye(4); I[-1,-1] = 0. # rank deficient matrix
    >>> matrix_rank(I)
    3
    >>> matrix_rank(np.zeros((4,4))) # All zeros - zero rank
    0
    >>> matrix_rank(np.ones((4,))) # 1 dimension - rank 1 unless all 0
    1
    >>> matrix_rank(np.zeros((4,)))
    0
    >>> matrix_rank([1]) # accepts array-like
    1

    Notes
    -----
    Golub and van Loan [1]_ define "numerical rank deficiency" as using
    tol=eps*S[0] (where S[0] is the maximum singular value and thus the
    2-norm of the matrix). This is one definition of rank deficiency,
    and the one we use here.  When floating point roundoff is the main
    concern, then "numerical rank deficiency" is a reasonable choice. In
    some cases you may prefer other definitions. The most useful measure
    of the tolerance depends on the operations you intend to use on your
    matrix. For example, if your data come from uncertain measurements
    with uncertainties greater than floating point epsilon, choosing a
    tolerance near that uncertainty may be preferable.  The tolerance
    may be absolute if the uncertainties are absolute rather than
    relative.

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, _Matrix Computations_.
    Baltimore: Johns Hopkins University Press, 1996.
    '''
    M = np.asarray(M)
    if M.ndim > 2:
        raise TypeError('array should have 2 or fewer dimensions')
    if M.ndim < 2:
        return int(not np.all(M==0))
    S = npl.svd(M, compute_uv=False)
    if tol is None:
        tol = S.max() * np.finfo(S.dtype).eps
    return np.sum(S > tol)


def rank(X, cond=1.0e-12):
    """
    Return the rank of a matrix X based on its generalized inverse,
    not the SVD.
    """
    X = np.asarray(X)
    if len(X.shape) == 2:
        D = npl.svd(X, compute_uv=False)
        return int(np.add.reduce(np.greater(D / D.max(), cond).astype(np.int32)))
    else:
        return int(not np.alltrue(np.equal(X, 0.)))


def fullrank(X, r=None):
    """
    Return a matrix whose column span is the same as X.

    If the rank of X is known it can be specified as r -- no check
    is made to ensure that this really is the rank of X.

    """

    if r is None:
        r = rank(X)

    V, D, U = npl.svd(X, full_matrices=0)
    order = np.argsort(D)
    order = order[::-1]
    value = []
    for i in range(r):
        value.append(V[:,order[i]])
    return np.asarray(np.transpose(value)).astype(np.float64)


class StepFunction(object):
    """
    A basic step function: values at the ends are handled in the simplest
    way possible: everything to the left of x[0] is set to ival; everything
    to the right of x[-1] is set to y[-1].

    Examples
    --------
    >>> from numpy import arange
    >>> from nipy.fixes.scipy.stats.models.utils import StepFunction
    >>>
    >>> x = arange(20)
    >>> y = arange(20)
    >>> f = StepFunction(x, y)
    >>>
    >>> print f(3.2)
    3.0
    >>> print f([[3.2,4.5],[24,-3.1]])
    [[  3.   4.]
     [ 19.   0.]]
    """

    def __init__(self, x, y, ival=0., sorted=False):

        _x = np.asarray(x)
        _y = np.asarray(y)

        if _x.shape != _y.shape:
            raise ValueError, 'in StepFunction: x and y do not have the same shape'
        if len(_x.shape) != 1:
            raise ValueError, 'in StepFunction: x and y must be 1-dimensional'

        self.x = np.hstack([[-np.inf], _x])
        self.y = np.hstack([[ival], _y])

        if not sorted:
            asort = np.argsort(self.x)
            self.x = np.take(self.x, asort, 0)
            self.y = np.take(self.y, asort, 0)
        self.n = self.x.shape[0]

    def __call__(self, time):

        tind = np.searchsorted(self.x, time) - 1
        _shape = tind.shape
        return self.y[tind]


def ECDF(values):
    """
    Return the ECDF of an array as a step function.
    """
    x = np.array(values, copy=True)
    x.sort()
    x.shape = np.product(x.shape,axis=0)
    n = x.shape[0]
    y = (np.arange(n) + 1.) / n
    return StepFunction(x, y)


def monotone_fn_inverter(fn, x, vectorized=True, **keywords):
    """
    Given a monotone function x (no checking is done to verify monotonicity)
    and a set of x values, return an linearly interpolated approximation
    to its inverse from its values on x.
    """

    if vectorized:
        y = fn(x, **keywords)
    else:
        y = []
        for _x in x:
            y.append(fn(_x, **keywords))
        y = np.array(y)

    a = np.argsort(y)

    return scipy.interpolate.interp1d(y[a], x[a])
