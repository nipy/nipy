""" Array utilities
"""
from __future__ import absolute_import
import numpy as np

def strides_from(shape, dtype, order='C'):
    """ Return strides as for continuous array shape `shape` and given `dtype`

    Parameters
    ----------
    shape : sequence
        shape of array to calculate strides from
    dtype : dtype-like
        dtype specifier for array
    order : {'C', 'F'}, optional
        whether array is C or FORTRAN ordered

    Returns
    -------
    strides : tuple
        seqence length ``len(shape)`` giving strides for continuous array with
        given `shape`, `dtype` and `order`

    Examples
    --------
    >>> strides_from((2,3,4), 'i4')
    (48, 16, 4)
    >>> strides_from((3,2), np.float)
    (16, 8)
    >>> strides_from((5,4,3), np.bool, order='F')
    (1, 5, 20)
    """
    dt = np.dtype(dtype)
    if dt.itemsize == 0:
        raise ValueError('Empty dtype "%s"' % dt)
    if order == 'F':
        strides = np.cumprod([dt.itemsize] + list(shape[:-1]))
    elif order == 'C':
        strides = np.cumprod([dt.itemsize] + list(shape)[::-1][:-1])
        strides = strides[::-1]
    else:
        raise ValueError('Unexpected order "%s"' % order)
    return tuple(strides)
