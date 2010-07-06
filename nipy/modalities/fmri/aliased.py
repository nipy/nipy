# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
An implementation of Functions in sympy that allow 'anonymous'
functions that can be evaluated when 'lambdified'. 
"""

import sympy


def lambdify(args, expr):
    """ Returns function for fast calculation of numerical values

    A modified version of sympy's lambdify that will find 'aliased'
    Functions and substitute them appropriately at evaluation time.

    See ``sympy.lambdify`` for more detail.

    Parameters
    ----------
    args : object or sequence of objects
       May well be sympy Symbols
    expr : expression
       The expression is anything that can be passed to the sympy
       lambdify function, meaning anything that gives valid code from
       ``str(expr)``.

    Examples
    --------
    >>> x = sympy.Symbol('x')
    >>> f = lambdify(x, x**2)
    >>> f(3)
    9
    """
    n = _imp_namespace(expr)
    # There was a bug in sympy such that dictionaries passed in as first
    # namespaces to lambdify, before modules, would get overwritten by
    # later calls to lambdify.  The next two lines are to get round this
    # bug.  
    from sympy.utilities.lambdify import _get_namespace
    np_ns = _get_namespace('numpy').copy()
    return sympy.lambdify(args, expr, modules=(n, np_ns))


def _imp_namespace(expr, namespace=None):
    """ Return namespace dict with function implementations

    We need to search for functions in anything that can be thrown at
    us - that is - anything that could be passed as `expr`.  Examples
    include sympy expressions, as well tuples, lists and dicts that may
    contain sympy expressions.

    Parameters
    ----------
    expr : object
       Something passed to lambdify, that will generate valid code from
       ``str(expr)``. 
    namespace : None or mapping
       Namespace to fill.  None results in new empty dict

    Returns
    -------
    namespace : dict
       dkct with keys of implemented function names within `expr` and
       corresponding values being the numerical implementation of
       function
    """
    if namespace is None:
        namespace = {}
    # tuples, lists, dicts are valid expressions
    if isinstance(expr, (list, tuple)):
        for arg in expr:
            _imp_namespace(arg, namespace)
        return namespace
    elif isinstance(expr, dict):
        for key, val in expr.items():
            # functions can be in dictionary keys
            _imp_namespace(key, namespace)
            _imp_namespace(val, namespace)
        return namespace
    # sympy expressions may be Functions themselves
    if hasattr(expr, 'func'):
        if (isinstance(expr.func, sympy.FunctionClass) and
            hasattr(expr.func, 'alias')):
            name = expr.func.__name__
            imp = expr.func.alias
            if name in namespace and namespace[name] != imp:
                raise ValueError('We found more than one '
                                 'implementation with name '
                                 '"%s"' % name)
            namespace[name] = imp
    # and / or they may take Functions as arguments
    if hasattr(expr, 'args'):
        for arg in expr.args:
            _imp_namespace(arg, namespace)
    return namespace


def aliased_function(symfunc, alias):
    """ Add implementation `alias` to symbolic function `symfunc`

    Parameters
    ----------
    symfunc : str or ``sympy.FunctionClass`` instance
       If str, then create new anonymous sympy function with this as
       name.  If `symfunc` is a sympy function, attach implementation to
       function
    alias : callable
       numerical implementation of function for use in ``lambdify``

    Returns
    -------
    afunc : sympy.FunctionClass instance
       sympy function with attached implementation
    """
    # if name, create anonymous function to hold alias
    if isinstance(symfunc, basestring):
        symfunc = sympy.FunctionClass(sympy.Function, symfunc)
    # We need to attach as a method because symfunc will be a class
    symfunc.alias = staticmethod(alias)
    return symfunc

