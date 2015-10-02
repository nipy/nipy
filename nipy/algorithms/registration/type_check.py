"""
Utilities to test whether a variable is of, or convertible to, a particular type
"""
import numpy as np


def _check_type(x, t):
    try:
        y = t(x)
        return True
    except:
        return False


def check_type(x, t, accept_none=False):
    if accept_none:
        if x is None:
            return
    if not _check_type(x, t):
        raise ValueError('Argument should be convertible to %s' % t)


def check_type_and_shape(x, t, s, accept_none=False):
    """
    x : array-like argument to be checked
    t : type of array values 
    s : length or array shape
    """	
    if accept_none:
        if x is None:
            return
    try:
        shape = (int(s), )
    except:
        shape = tuple(s)
    try:
        y = np.asarray(x)
        ok_type = _check_type(y[0], t)
        ok_shape = (y.shape == shape)
    except:
        raise ValueError('Argument should be convertible to ndarray')
    if not ok_type:
        raise ValueError('Array values should be convertible to %s' % t)
    if not ok_shape:
        raise ValueError('Array shape should be equivalent to %s' % shape)

