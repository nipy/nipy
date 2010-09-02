# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# add-ons to scipy.optimize

import numpy as np 
from scipy.optimize import brent, approx_fprime

_STEP = np.sqrt(np.finfo(float).eps)


def _linesearch_brent(func, p, xi, tol=1e-3):
    """Line-search algorithm using Brent's method.

    Find the minimium of the function ``func(x0+ alpha*direc)``.
    """
    def myfunc(alpha):
        return func(p + alpha * xi)
    alpha_min, fret, iter, num = brent(myfunc, full_output=1, tol=tol)
    xi = alpha_min*xi
    return np.squeeze(fret), p+xi


def _wrap(function, args):
    ncalls = [0]
    def wrapper(x):
        ncalls[0] += 1
        return function(x, *args)
    return ncalls, wrapper


def fmin_steepest(f, x0, fprime=None, xtol=1e-4, ftol=1e-4, 
                  step=_STEP,
                  maxiter=None, callback=None): 

    x = np.asarray(x0).flatten()
    fval = np.squeeze(f(x))
    it = 0 
    if maxiter == None: 
        maxiter = x.size*1000
    if fprime == None:
        grad_calls, myfprime = _wrap(approx_fprime, (f, step))
    else:
        grad_calls, myfprime = _wrap(fprime, args)

    while it < maxiter:
        it = it + 1
        x0 = x 
        fval0 = fval
        print('Computing gradient...')
        direc = myfprime(x)
        direc = direc / np.sqrt(np.sum(direc**2))
        print('Performing line search...')
        fval, x = _linesearch_brent(f, x, direc, tol=xtol*100)
        if not callback == None:
            callback(x)
        if (2.0*(fval0-fval) <= ftol*(abs(fval0)+abs(fval))+1e-20): 
            break
        
    print('Number of iterations: %d' % it)
    print('Minimum criterion value: %f' % fval)

    return x 


