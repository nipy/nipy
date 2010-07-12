# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" This module defines some convenience functions of time.

interp : an expresion for a interpolated function of time

linear_interp : an expression for a linearly interpolated function of
   time

step_function : an expression for a step function of time

events : a convenience function to generate sums of events

blocks : a convenience function to generate sums of blocks

convolve_functions : numerically convolve two functions of time

fourier_basis : a convenience function to generate a Fourier basis
"""

import itertools

import numpy as np
import numpy.fft as FFT
from scipy.interpolate import interp1d

from sympy import DiracDelta, Symbol 
from sympy import sin as sympy_sin
from sympy import cos as sympy_cos
from sympy import pi as sympy_pi

from . import formula
from . import aliased
from .aliased import aliased_function, lambdify

t = formula.Term('t')


def fourier_basis(freq):
    """ sin and cos Formula for Fourier drift

    The Fourier basis consists of sine and cosine waves of given
    frequencies.

    Parameters
    ----------
    freq : sequence of float
        Frequencies for the terms in the Fourier basis.

    Returns
    -------
    f : Formula

    Examples
    --------
    >>> f=fourier_basis([1,2,3])
    >>> f.terms
    array([cos(2*pi*t), sin(2*pi*t), cos(4*pi*t), sin(4*pi*t), cos(6*pi*t),
           sin(6*pi*t)], dtype=object)
    >>> f.mean
    _b0*cos(2*pi*t) + _b1*sin(2*pi*t) + _b2*cos(4*pi*t) + _b3*sin(4*pi*t) + _b4*cos(6*pi*t) + _b5*sin(6*pi*t)
    """
    r = []
    for f in freq:
        r += [sympy_cos((2*sympy_pi*f*t)),
              sympy_sin((2*sympy_pi*f*t))]
    return formula.Formula(r)


def interp(times, values, fill=0, name=None, **kw):
    """ Generic interpolation function of t given `times` and `values`

    Imterpolator such that:
    
    f(times[i]) = values[i]

    if t < times[0]:
        f(t) = fill

    See ``scipy.interpolate.interp1d`` for details of interpolation
    types and other keyword arguments.  Default is 'kind' is linear,
    making this function, by default, have the same behavior as
    ``linear_interp``. 

    Parameters
    ----------
    times : array-like
        Increasing sequence of times
    values : array-like
        Values at the specified times
    fill : float, optional
        Value on the interval (-np.inf, times[0]). Default 0.
    name : None or str, optional
        Name of symbolic expression to use. If None, a default is used.
    **kw : keyword args, optional
        passed to ``interp1d``
        
    Returns
    -------
    f : sympy expression 
        A Function of t.

    Examples
    --------
    >>> s = interp([0,4,5.],[2.,4,6], bounds_error=False)
    >>> tval = np.array([-0.1,0.1,3.9,4.1,5.1])
    >>> res = aliased.lambdify(t, s)(tval)
    >>> # nans outside bounds
    >>> np.isnan(res)
    array([ True, False, False, False,  True], dtype=bool)
    >>> # interpolated values otherwise
    >>> np.allclose(res[1:-1], [2.05, 3.95, 4.2])
    True
    """
    interpolator = interp1d(times, values, **kw)
    # make a new name if none provided
    if name is None:
        name = 'interp%d' % interp.counter
        interp.counter += 1
    s = aliased_function(name, interpolator)
    return s(t)

interp.counter = 0


def linear_interp(times, values, fill=0, name=None, **kw):
    """ Linear interpolation function of t given `times` and `values`

    Imterpolator such that:
    
    f(times[i]) = values[i]

    if t < times[0]:
        f(t) = fill

    This version of the function enforces the 'linear' kind of
    interpolation (argument to ``scipy.interpolate.interp1d``). 

    Parameters
    ----------
    times : array-like
        Increasing sequence of times
    values : array-like
        Values at the specified times
    fill : float, optional
        Value on the interval (-np.inf, times[0]). Default 0.
    name : None or str, optional
        Name of symbolic expression to use. If None, a default is used.
    **kw : keyword args, optional
        passed to ``interp1d``
        
    Returns
    -------
    f : sympy expression 
        A Function of t.

    Examples
    --------
    >>> s = linear_interp([0,4,5.],[2.,4,6], bounds_error=False)
    >>> tval = np.array([-0.1,0.1,3.9,4.1,5.1])
    >>> res = aliased.lambdify(t, s)(tval)
    >>> # nans outside bounds
    >>> np.isnan(res)
    array([ True, False, False, False,  True], dtype=bool)
    >>> # interpolated values otherwise
    >>> np.allclose(res[1:-1], [2.05, 3.95, 4.2])
    True
    """
    kind = kw.get('kind')
    if kind is None:
        kw['kind'] = 'linear'
    elif kind != 'linear':
        raise ValueError('Only linear interpolation supported')
    return interp(times, values, fill, name, **kw)


def step_function(times, values, name=None, fill=0):
    """ Right-continuous step function of time t

    Function of t such that

    f(times[i]) = values[i]

    if t < times[0]:
        f(t) = fill

    Parameters
    ----------
    times : (N,) sequence
       Increasing sequence of times
    values : (N,) sequence
       Values at the specified times
    fill : float
       Value on the interval (-np.inf, times[0])
    name : str
       Name of symbolic expression to use. If None, a default is used.

    Returns
    -------
    f_t : sympy expr
       Sympy expression f(t) where f is a sympy implemented anonymous
       function of time that implements the step function.  To get
       the numerical version of the function, use ``lambdify(t, f_t)``

    Examples
    --------
    >>> s = step_function([0,4,5],[2,4,6])
    >>> tval = np.array([-0.1,3.9,4.1,5.1])
    >>> lam = aliased.lambdify(t, s)
    >>> lam(tval)
    array([ 0.,  2.,  4.,  6.])
    """
    if name is None:
        name = 'step%d' % step_function.counter
        step_function.counter += 1
    
    def _imp(x):
        x = np.asarray(x)
        f = np.zeros(x.shape) + fill
        for time, val in zip(times, values):
            f[x >= time] = val
        return f

    s = aliased_function(name, _imp)
    return s(t)

# Initialize counter for step function
step_function.counter = 0


def events(times, amplitudes=None, f=DiracDelta, g=Symbol('a')):
    """ Return a sum of functions based on a sequence of times.

    Parameters
    ----------
    times : sequence
       vector of onsets length $N$
    amplitudes : None or sequence length $N$, optional
       Optional sequence of amplitudes. None (default) results in
       sequence length $N$ of 1s
    f : sympy.Function, optional
       Optional function. Defaults to DiracDelta, can be replaced with
       another function, f, in which case the result is the convolution
       with f.
    g : sympy.Basic, optional
       Optional sympy expression function of amplitudes.  The
       amplitudes, should be represented by the symbol 'a', which will
       be substituted, by the corresponding value in `amplitudes`.

    Returns
    -------
    sum_expression : Sympy.Add
       Sympy expression of time $t$, where onsets, as a function of $t$,
       have been symbolically convolved with function `f`, and any
       function `g` of corresponding amplitudes.

    Examples
    --------
    We import some sympy stuff so we can test if we've got what we
    expected

    >>> from sympy import DiracDelta, Symbol, Function
    >>> from nipy.modalities.fmri.formula import Term
    >>> t = Term('t')
    
    >>> evs = events([3,6,9])
    >>> evs == DiracDelta(-9 + t) + DiracDelta(-6 + t) + DiracDelta(-3 + t)
    True
    >>> hrf = Function('hrf')
    >>> evs = events([3,6,9], f=hrf)
    >>> evs == hrf(-9 + t) + hrf(-6 + t) + hrf(-3 + t)
    True
    >>> evs = events([3,6,9], amplitudes=[2,1,-1])
    >>> evs == -DiracDelta(-9 + t) + 2*DiracDelta(-3 + t) + DiracDelta(-6 + t)
    True
    """
    e = 0
    asymb = Symbol('a')
    if amplitudes is None:
        amplitudes = itertools.cycle([1])
    for _t, a in zip(times, amplitudes):
        e = e + g.subs(asymb, a) * f(t-_t)
    return e


def blocks(intervals, amplitudes=None):
    """ Step function based on a sequence of intervals.

    Parameters
    ----------
    intervals : (S,) sequence of (2,) sequences
       Sequence (S0, S1, ... S(N-1)) of sequences, where S0 (etc) are
       sequences of length 2, giving 'on' and 'off' times of block
    amplitudes : (S,) sequence of float, optional
       Optional amplitudes for each block. Defaults to 1.

    Returns
    -------
    b_of_t : sympy expr
       Sympy expression b(t) where b is a sympy anonymous function of
       time that implements the block step function

    Examples
    --------
    >>> on_off = [[1,2],[3,4]]
    >>> tval = np.array([0.4,1.4,2.4,3.4])
    >>> b = blocks(on_off)
    >>> lam = aliased.lambdify(t, b)
    >>> lam(tval)
    array([ 0.,  1.,  0.,  1.])
    >>> b = blocks(on_off, amplitudes=[3,5])
    >>> lam = aliased.lambdify(t, b)
    >>> lam(tval)
    array([ 0.,  3.,  0.,  5.])
    """
    t = [-np.inf]
    v = [0]
    if amplitudes is None:
        amplitudes = itertools.cycle([1])
    for _t, a in zip(intervals, amplitudes):
        t += list(_t)
        v += [a, 0]
    t.append(np.inf)
    v.append(0)
    return step_function(t, v)


def convolve_functions(fn1, fn2, interval, dt, padding_f=0.1, name=None):
    """ Expression containing numerical convolution of `fn1` with `fn2`
    
    Parameters
    ----------
    fn1 : sympy expr
       An expression that is a function of t only.
    fn2 : sympy expr
       An expression that is a function of t only.
    interval : (2,) sequence of float
       The start and end of the interval over which to convolve the two
       functions.
    dt : float
       Time step for discretization.  We use this for creating the
       interpolator to form the numerical implementation
    padding_f : float, optional
       Padding fraction added to the left and right in the convolution.
       Padding value is fraction of the length given by `interval`
    name : None or str, optional
       Name of the convolved function in the resulting expression. 
       Defaults to one created by ``utils.interp``.
            
    Returns
    -------
    f : sympy expr
       An expression that is a function of t only.

    Examples
    --------
    >>> import sympy
    >>> t = sympy.Symbol('t')
    >>> # This is a square wave on [0,1]
    >>> f1 = (t > 0) * (t < 1)

    The convolution of ``f1`` with itself is a triangular wave on [0,2],
    peaking at 1 with height 1
    
    >>> tri = convolve_functions(f1, f1, [0,2], 1.0e-3, name='conv')
    >>> print tri
    conv(t)

    Get the numerical values for a time vector
    
    >>> ftri = aliased.lambdify(t, tri)
    >>> x = np.linspace(0,2,11)
    >>> y = ftri(x)

    The peak is at 1

    >>> x[np.argmax(y)]
    1.0
    """
    # Note that - from the doctest above - y is
    """
    array([ -3.90255908e-16,   1.99000000e-01,   3.99000000e-01,
           5.99000000e-01,   7.99000000e-01,   9.99000000e-01,
           7.99000000e-01,   5.99000000e-01,   3.99000000e-01,
           1.99000000e-01,   6.74679706e-16])
    """
    # - so he peak value is 1-dt - rather than 1 - but we get the same
    # result from using np.convolve - see tests.
    mn_i, mx_i = sorted(interval)
    pad_t = (mx_i - mn_i) * padding_f
    time = np.arange(mn_i, mx_i + pad_t, dt)
    # get values at times from expressions
    f1 = lambdify(t, fn1)
    f2 = lambdify(t, fn2)
    _fn1 = np.atleast_1d(f1(time))
    _fn2 = np.atleast_1d(f2(time))
    # do convolution
    _fft1 = FFT.rfft(_fn1)
    _fft2 = FFT.rfft(_fn2)
    value = FFT.irfft(_fft1 * _fft2) * dt
    assert value.ndim == 1
    _minshape = min(time.shape[0], value.shape[0])
    time = time[0:_minshape]
    value = value[0:_minshape]
    return interp(time, value, bounds_error=False, name=name)
