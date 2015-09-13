""" Collection of utility functions and classes

Some of these come from the matplotlib ``cbook`` module with thanks.
"""

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
