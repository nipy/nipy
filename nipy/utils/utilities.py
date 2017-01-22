""" Collection of utility functions and classes

Some of these come from the matplotlib ``cbook`` module with thanks.
"""

from functools import reduce
from operator import mul


def is_iterable(obj):
    """ Return True if `obj` is iterable
    """
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def is_numlike(obj):
    """ Return True if `obj` looks like a number
    """
    try:
        obj + 1
    except:
        return False
    return True


def seq_prod(seq, initial=1):
    """ General product of sequence elements

    Parameters
    ----------
    seq : sequence
        Iterable of values to multiply.
    initial : object, optional
        Initial value

    Returns
    -------
    prod : object
        Result of ``initial * seq[0] * seq[1] .. ``.
    """
    return reduce(mul, seq, initial)
