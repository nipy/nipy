"""
This module defines some convenience functions of time.

linear_interp : a Formula for a linearly interpolated function of time

step_function : a Formula for a step function of time

events : a convenience function to generate sums of events

blocks : a convenience function to generate sums of blocks

convolve_functions : numerically convolve two functions of time
"""

__docformat__ = 'restructuredtext'

import numpy as np
import numpy.fft as FFT
from scipy.interpolate import interp1d

from sympy import Function, DiracDelta, Symbol
from formula import Formula, Term, Design

t = Term('t')


def linear_interp(times, values, fill=0, name=None, **kw):
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
        
    name : str
        Name of symbolic expression to use. If None,
        a default is used.

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

    if name is None:
        name = 'interp%d' % linear_interp.counter
        linear_interp.counter += 1

    s = Symbol(name)
    ff = Formula([s(t)])
    ff.aliases[name] = i
    return ff
linear_interp.counter = 0

def step_function(times, values, name=None, fill=0):
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
        
    name : str
        Name of symbolic expression to use. If None,
        a default is used.

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

    if name is None:
        name = 'step%d' % step_function.counter
        step_function.counter += 1

    s = Symbol(name)
    ff = Formula([s(t)])
    ff.aliases[name] = anon
    return ff
step_function.counter = 0

def events(times, amplitudes=None, f=DiracDelta, g=Symbol('a')):
    """
    Return a sum of functions
    based on a sequence of times.

    Inputs:
    =======

    times : [float]

    amplitudes : [float]
        Optional sequence of amplitudes. Default to 1.

    f : sympy.Function
        Optional function. Defaults to DiracDelta, can be replaced with
        another function, f, in which case the result is the convolution
        with f.

    g : sympy.Basic
        Optional sympy expression function involving 'a', which
        will be substituted by the values of in the generator.

    Examples:
    =========

    >>> events([3,6,9])
    DiracDelta(-9 + t) + DiracDelta(-6 + t) + DiracDelta(-3 + t)
    >>> h = Symbol('hrf')
    >>> events([3,6,9], f=h)
    hrf(-9 + t) + hrf(-6 + t) + hrf(-3 + t)
    >>>

    >>> events([3,6,9], amplitudes=[2,1,-1])
    -DiracDelta(-9 + t) + 2*DiracDelta(-3 + t) + DiracDelta(-6 + t)

    >>> b = [Symbol('b%d' % i, dummy=True) for i in range(3)]
    >>> a = Symbol('a')
    >>> p = b[0] + b[1]*a + b[2]*a**2
    >>> events([3,6,9], amplitudes=[2,1,-1], g=p)
    (2*_b1 + 4*_b2 + _b0)*DiracDelta(-3 + t) + (-_b1 + _b0 + _b2)*DiracDelta(-9 + t) + (_b0 + _b1 + _b2)*DiracDelta(-6 + t)

    >>> h = Symbol('hrf')
    >>> events([3,6,9], amplitudes=[2,1,-1], g=p, f=h)
    (2*_b1 + 4*_b2 + _b0)*hrf(-3 + t) + (-_b1 + _b0 + _b2)*hrf(-9 + t) + (_b0 + _b1 + _b2)*hrf(-6 + t)

    """
    e = 0
    asymb = Symbol('a')

    if amplitudes is None:
        def _amplitudes():
            while True:
                yield 1
        amplitudes = _amplitudes()

    for _t, a in zip(times, amplitudes):
        e = e + g.subs(asymb, a) * f(t-_t)
    return e

def blocks(intervals, amplitudes=None, g=Symbol('a')):
    """
    Return a step function
    based on a sequence of intervals.

    Inputs:
    =======

    intervals : [(float, float)]
        "On" intervals for the block.

    amplitudes : [float]
        Optional amplitudes for each block. Defaults to 1.

    g : sympy.Basic
        Optional sympy expression function involving 'a', which
        will be substituted for 'a' in the generator.

    Examples:
    =========
    
    >>> tval = np.array([0.4,1.4,2.4,3.4]).view(np.dtype([('t', np.float)]))
    >>> b = blocks([[1,2],[3,4]])
    >>> from formula import Design
    >>> d = Design(b)
    >>> d(tval)
    array([(0.0,), (1.0,), (0.0,), (1.0,)], 
          dtype=[('step0(t)', '<f8')])

    >>> b = blocks([[1,2],[3,4]], amplitudes=[3,5])
    >>> d = Design(b)
    >>> d(tval)
    array([(0.0,), (3.0,), (0.0,), (5.0,)], 
          dtype=[('step1(t)', '<f8')])

    >>> a = Symbol('a')
    >>> b = blocks([[1,2],[3,4]], amplitudes=[3,5], g=a+1)
    >>> d = Design(b)
    >>> d(tval)
    array([(0.0,), (4.0,), (0.0,), (6.0,)], 
          dtype=[('step2(t)', '<f8')])

    """
    t = [-np.inf]
    v = [0]
    asymb = Symbol('a')
    if amplitudes is None:
        def _amplitudes():
            while True:
                yield 1
        amplitudes = _amplitudes()

    for _t, a in zip(intervals, amplitudes):
        t += list(_t)
        v += [g.subs(asymb, a), 0]

    t.append(np.inf)
    v.append(0)

    return step_function(t, v)

def convolve_functions(fn1, fn2, interval, dt, padding_f=0.1):
    """
    Convolve fn1 with fn2.
    
    :Parameters:
        `fn1` : TODO
            TODO
        `fn2` : TODO
            TODO
        `interval` : TODO
            TODO
        `dt` : TODO
            TODO
        `padding_f` : float
            TODO
            
    :Returns: TODO
    """

    max_interval, min_interval = max(interval), min(interval)
    ltime = max_interval - min_interval
    time = np.arange(min_interval, max_interval + padding_f * ltime, dt)

    _fn1 = np.array(Vectorize(fn1)(time))
    _fn2 = np.array(Vectorize(fn2)(time))

    _fft1 = FFT.rfft(_fn1)
    _fft2 = FFT.rfft(_fn2)

    value = FFT.irfft(_fft1 * _fft2)
    _minshape = min(time.shape[0], value.shape[-1])
    time = time[0:_minshape]
    value = value[0:_minshape]

    l = linear_interp(time + min_interval, value)
    print l

class Vectorize(Design):
    """
    This class can be used to take a (single-valued) sympy
    expression with only 't' as a Symbol and return a 
    callable that can be evaluated at an array of floats.

    Inputs:
    =======

    expr : sympy.Basic or Formula
        Expression with 't' the only Symbol. If it is a 
        Formula, then the only unknown symbol (besides 
        the coefficients) should be 't'.

    """

    def __init__(self, expr):
        if not isinstance(expr, Formula):
            expr = Formula([expr])
        Design.__init__(self, expr, return_float=True)

    def __call__(self, t):
        t = np.asarray(t).astype(np.float)
        tval = t.view(np.dtype([('t', np.float)]))
        return Design.__call__(self, tval)
