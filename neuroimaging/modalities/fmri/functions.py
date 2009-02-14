"""
This module defines some convenience functions of time.

step_function : a Formula for a step function of time

linear_interp : a Formula for a linearly interpolated function of time

events : a convenience function to generate sums of events

events_and_amplitudes : a convenience function to generate sums of events,
                        paired with amplitudes
                 
"""

__docformat__ = 'restructuredtext'

import numpy as np

from scipy.interpolate import interp1d

from sympy import Function, DiracDelta, Symbol
from formula import Formula, Term

t = Term('t')

def linear_interp(times, values, fill=0, **kw):
    """
    Linear interpolation function such that

    f(times[i]) = values[i]

    if t < times[0]:
        f(t) = fill

    Inputs:
    =======

    times : ndarray
        Increasing sequence of times

    values : ndarray
        Values at the specified times

    fill : float
        Value on the interval (-np.inf, times[0])
        
    Outputs:
    ========

    f : Formula
        A Formula with only a linear interpolator, as a function of t.

    Examples:
    =========

    >>> s=linear_interp([0,4,5.],[2.,4,6], bounds_error=False)
    >>> from formula import Design
    >>> d=Design(s)
    >>> tval = np.array([-0.1,0.1,3.9,4.1,5.1]).view(np.dtype([('t', np.float)]))
    >>> d(tval)
    array([(nan,), (2.0499999999999998,), (3.9500000000000002,),
           (4.1999999999999993,), (nan,)],
          dtype=[('interp0(t)', '<f8')])

    """
    kw['kind'] = 'linear'
    i = interp1d(times, values, **kw)
    s = Function('interp%d' % linear_interp.counter, dummy=True)
    linear_interp.counter += 1
    ff = Formula([s(t)])
    ff['interp'] = i
    return ff
linear_interp.counter = 0

def step_function(times, values, fill=0):
    """
    Right-continuous step function such that

    f(times[i]) = values[i]

    if t < times[0]:
        f(t) = fill

    Inputs:
    =======

    times : ndarray
        Increasing sequence of times

    values : ndarray
        Values at the specified times

    fill : float
        Value on the interval (-np.inf, times[0])
        
    Outputs:
    ========

    f : Formula
        A Formula with only a step function, as a function of t.

    Examples:
    =========

    >>> s=step_function([0,4,5],[2,4,6])
    >>> tval = np.array([-0.1,3.9,4.1,5.1]).view(np.dtype([('t', np.float)]))
    >>> from formula import Design
    >>> d=Design(s)
    >>> d(tval)
    array([(0.0,), (2.0,), (4.0,), (6.0,)],
          dtype=[('step0(t)', '<f8')])
    >>>

    """
    times = np.asarray(times)
    values = np.asarray(values)        

    def anon(x, times=times, values=values, fill=fill):
        d = values[1:] - values[:-1]
        f = np.less(x, times[0]) * fill + np.greater(x, times[0]) * values[0]
        for i in range(d.shape[0]):
            f = f + np.greater(x, times[i+1]) * d[i]
        return f
    s = Symbol('step%d' % step_function.counter, dummy=True)
    step_function.counter += 1
    ff = Formula([s(t)])
    ff['step'] = anon
    return ff
step_function.counter = 0

def events(generator, f=DiracDelta):
    """
    Return a sum of functions
    based on a generator of times.

    Inputs:
    =======

    generator : [float]

    f : sympy.Function
        Optional function. Defaults to DiracDelta, can be replaced with
        another function, f, in which case the result is the convolution
        with f.

    Examples:
    =========

    >>> events([3,6,9])
    DiracDelta(-9 + t) + DiracDelta(-6 + t) + DiracDelta(-3 + t)
    >>> h = Symbol('hrf')
    >>> events([3,6,9], f=h)
    hrf(-9 + t) + hrf(-6 + t) + hrf(-3 + t)
    >>>

    """
    e = 0
    for _t in generator:
        e = e + f(t-_t)
    return e

def events_and_amplitudes(generator, f=DiracDelta, g=None):
    """
    Return a sum of DiracDelta functions (optionally convolved with f)
    based on a generator of times.

    Inputs:
    =======

    generator : [(float, float)]

    f : sympy.Function
        Optional function, can be replaced with HRF. Defaults to DiracDelta

    g : sympy.Basic
        Optional sympy expression function involving 'a', which
        will be substituted by the values of in the generator.

    Examples:
    =========

    >>> events_and_amplitudes(np.array([[2,3],[1,6],[-1,9]]))
    -DiracDelta(-9 + t) + 2*DiracDelta(-3 + t) + DiracDelta(-6 + t)
    >>> from sympy import Symbol
    >>> b = [Symbol('b%d' % i, dummy=True) for i in range(3)]
    >>> a = Symbol('a')
    >>> p = b[0] + b[1]*a + b[2]*a**2
    >>> events_and_amplitudes(np.array([[2,3],[1,6],[-1,9]]), g=p)
    (2*_b1 + 4*_b2 + _b0)*DiracDelta(-3 + t) + (-_b1 + _b0 + _b2)*DiracDelta(-9 + t) + (_b0 + _b1 + _b2)*DiracDelta(-6 + t)
    >>>
    >>> h = Symbol('hrf')
    >>> events_and_amplitudes(np.array([[2,3],[1,6],[-1,9]]), g=p, f=h)
    (2*_b1 + 4*_b2 + _b0)*hrf(-3 + t) + (-_b1 + _b0 + _b2)*hrf(-9 + t) + (_b0 + _b1 + _b2)*hrf(-6 + t)
    
    """
    e = 0
    asymb = Symbol('a')
    if g is None:
        for a, _t in generator:
            e = e + a * f(t-_t)
    else:
        for a, _t in generator:
            e = e + g.subs(asymb, a) * f(t-_t)
    return e

def blocks(generator):
    """
    Return a step function
    based on a generator of times.

    Inputs:
    =======

    generator : [(float, float)]

    f : sympy.Function
        Optional function, can be replaced with HRF. Defaults to DiracDelta

    g : sympy.Basic
        Optional sympy expression function involving 'a', which
        will be substituted by the values of in the generator.

    Examples:
    =========
    
    """
    e = 0
    asymb = Symbol('a')
    if g is None:
        for a, _t in generator:
            e = e + a * f(t-_t)
    else:
        for a, _t in generator:
            e = e + g.subs(asymb, a) * f(t-_t)
    return e

