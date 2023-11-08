""" Array utilities
"""
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
        sequence length ``len(shape)`` giving strides for continuous array with
        given `shape`, `dtype` and `order`

    Examples
    --------
    >>> strides_from((2,3,4), 'i4')
    (48, 16, 4)
    >>> strides_from((3,2), np.float64)
    (16, 8)
    >>> strides_from((5,4,3), np.bool_, order='F')
    (1, 5, 20)
    """
    dt = np.dtype(dtype)
    if dt.itemsize == 0:
        raise ValueError(f'Empty dtype "{dt}"')
    if order == 'F':
        strides = np.cumprod([dt.itemsize] + list(shape[:-1]))
    elif order == 'C':
        strides = np.cumprod([dt.itemsize] + list(shape)[::-1][:-1])
        strides = strides[::-1]
    else:
        raise ValueError(f'Unexpected order "{order}"')
    return tuple(strides)
