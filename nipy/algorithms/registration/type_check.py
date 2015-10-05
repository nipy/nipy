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
    """
    Checks whether a variable is convertible to a certain type.
    A ValueError is raised if test fails.

    Parameters
    ----------
    x : object
        Input argument to be checked.
    t : type
        Target type.
    accept_none : bool
        If True, skip errors if `x` is None.
    """	
    if accept_none:
        if x is None:
            return
    if not _check_type(x, t):
        raise ValueError('Argument should be convertible to %s' % t)


def check_type_and_shape(x, t, s, accept_none=False):
    """
    Checks whether a sequence is convertible to a numpy ndarray with
    given shape, and if the elements are convertible to a certain type.
    A ValueError is raised if test fails.

    Parameters
    ----------
    x : sequence
        Input sequence to be checked.
    t : type
        Target element-wise type.
    s : sequence of ints
        Target shape.
    accept_none : bool
        If True, skip errors if `x` is None.
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

