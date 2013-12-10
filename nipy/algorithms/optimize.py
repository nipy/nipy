# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# add-ons to scipy.optimize

import numpy as np 
from scipy.optimize import brent, approx_fprime


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
                  maxiter=None, epsilon=1.4901161193847656e-08,
                  callback=None, disp=True):
    """
    Minimize a function using a steepest gradient descent
    algorithm. This complements the collection of minimization
    routines provided in scipy.optimize. Steepest gradient iterations
    are cheaper than in the conjugate gradient or Newton methods,
    hence convergence may sometimes turn out faster algthough more
    iterations are typically needed.

    Parameters
    ----------
    f : callable
      Function to be minimized
    x0 : array
      Starting point
    fprime : callable
      Function that computes the gradient of f
    xtol : float
      Relative tolerance on step sizes in line searches
    ftol : float
      Relative tolerance on function variations
    maxiter : int
      Maximum number of iterations
    epsilon : float or ndarray
      If fprime is approximated, use this value for the step
    size (can be scalar or vector).
    callback : callable
      Optional function called after each iteration is complete
    disp : bool
      Print convergence message if True

    Returns
    -------
    x : array
      Gradient descent fix point, local minimizer of f
    """
    x = np.asarray(x0).flatten()
    fval = np.squeeze(f(x))
    it = 0 
    if maxiter == None: 
        maxiter = x.size*1000
    if fprime == None:
        grad_calls, myfprime = _wrap(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = _wrap(fprime, args)

    while it < maxiter:
        it = it + 1
        x0 = x 
        fval0 = fval
        if disp:
            print('Computing gradient...')
        direc = myfprime(x)
        direc = direc / np.sqrt(np.sum(direc**2))
        if disp:
            print('Performing line search...')
        fval, x = _linesearch_brent(f, x, direc, tol=xtol)
        if not callback == None:
            callback(x)
        if (2.0*(fval0-fval) <= ftol*(abs(fval0)+abs(fval))+1e-20): 
            break
        
        if disp:
            print('Number of iterations: %d' % it)
            print('Minimum criterion value: %f' % fval)

    return x 


