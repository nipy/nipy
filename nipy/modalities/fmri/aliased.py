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
        if isinstance(expr, sympy.FunctionClass):
            # NNB t is undefined at this point
            expr = expr(t)
        n = {} 
        _add_aliases_to_namespace(n, expr)
        self.n = n.copy()
        from sympy.utilities.lambdify import _get_namespace
        for k, v in  _get_namespace('numpy').items():
            self.n[k] = v
        self._f = sympy.lambdify(args, expr, self.n)
        self.expr = expr
        
    def __call__(self, _t):
        return self._f(_t)


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


class AliasedFunctionClass(sympy.FunctionClass):
    """ 'anonymous' sympy functions

    Functions that can be replaed with an appropriate callable function
    when lambdifying.

    No checking is done on the signature of the alias.

    This is not meant to be called by users, rather
    use 'aliased_function'.
    """
    def __new__(cls, arg1, arg2, arg3=None, alias=None):
        r = sympy.FunctionClass.__new__(cls, arg1, arg2, arg3)
        if alias is not None:
            r.alias = new.instancemethod(lambda self, x: alias(x), r, cls)
        return r


def _add_aliases_to_namespace(namespace, *exprs):
    """
    Given a sequence of sympy expressions,
    find all aliases in each expression and add them to the namespace.
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


def aliased_function(symbol, alias):
    """ Create aliased function with `symbol` and `alias`

    Parameters
    ----------
    symbol : ``sympy.Symbol`` instance

    alias : God alone knows
    """
    y = AliasedFunctionClass(sympy.Function, symbol, alias=alias)
    return y

