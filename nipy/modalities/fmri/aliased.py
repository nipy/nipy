# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
An implementation of Functions in sympy that allow 'anonymous'
functions that can be evaluated when 'lambdified'. 

"""
import new
import warnings
import sympy


class lambdify(object):
    """
    A modified version of sympy's lambdify that
    will find 'aliased' Functions and substitute
    them appropriately at evaluation time.
    """
    def __init__(self, args, expr):
        ''' Initialize lambdifier

        This lambdifier only allows there to be one input argument, in
        fact, because of the __call__ signature.

        Parameters
        ----------
        args : object or sequence of objects
           May well be sympy Symbols
        expr : expression

        Examples
        --------
        >>> x = sympy.Symbol('x')
        >>> f = lambdify(x, x**2)
        >>> f(3)
        9
        '''
        n = {} 
        _add_aliases_to_namespace(n, expr)
        self.n = n.copy()
        from sympy.utilities.lambdify import _get_namespace
        for k, v in  _get_namespace('numpy').items():
            self.n[k] = v
        self._f = sympy.lambdify(args, expr, self.n)
        self.expr = expr
        
    def __call__(self, *args, **kwargs):
        return self._f(*args, **kwargs)


class vectorize(lambdify):
    """
    This class can be used to take a (single-valued) sympy
    expression with only 't' as a Symbol and return a 
    callable that can be evaluated at an array of floats.

    Parameters
    ----------
    expr : sympy expr
        Expression with 't' the only Symbol. If it is 
        an instance of sympy.FunctionClass, 
        then vectorize expr(t) instead.

    Returns
    -------
    f : callable
        A function that can be evaluated at an array of time points.
    """
    def __init__(self, expr):
        deft = sympy.DeferredVector('t')
        t = sympy.Symbol('t')
        if isinstance(expr, sympy.FunctionClass):
            expr = expr(t)
        lambdify.__init__(self, deft, expr.subs(t, deft))


def _add_aliases_to_namespace(namespace, *exprs):
    """ add aliases in sympy `exprs` to namespace `namespace`.
    """
    for expr in exprs:
        if hasattr(expr, 'alias') and isinstance(expr, sympy.FunctionClass):
            if namespace.has_key(str(expr)):
                if namespace[str(expr)] != expr.alias:
                    warnings.warn('two aliases with the same name were found')
            namespace[str(expr)] = expr.alias
        if hasattr(expr, 'func'):
            if (isinstance(expr.func, sympy.FunctionClass) and
                hasattr(expr.func, 'alias')):
                if namespace.has_key(expr.func.__name__):
                    if namespace[expr.func.__name__] != expr.func.alias:
                        warnings.warn('two aliases with the same name were found')
                namespace[expr.func.__name__] = expr.func.alias
        if hasattr(expr, 'args'):
            try:
                _add_aliases_to_namespace(namespace, *expr.args)
            except TypeError:
                pass
    return namespace


def aliased_function(symfunc, alias):
    """ Add implementation `alias` to symbolic function `symfunc`

    Parameters
    ----------
    symfunc : str or ``sympy.FunctionClass`` instance
       If str, then create new anonymous sympy function with this name.
       If `symfunc` is a sympy function, attach implementation to
       function
    alias : callable
       numerical implementation of function for use in ``lambdify``

    Returns
    -------
    afunc : sympy.FunctionClass instance
       function with attached implementation
    """
    # if name, create anonymous function to hold alias
    if isinstance(symfunc, basestring):
        symfunc = sympy.FunctionClass(sympy.Function, symfunc)
    # We need to attach as a method because symfunc will be a class
    symfunc.alias = new.instancemethod(lambda self, *args: alias(*args),
                                       symfunc, symfunc.__class__)
    return symfunc

