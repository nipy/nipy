"""
This module defines some convenience functions of time.

linear_interp : a Formula for a linearly interpolated function of time

step_function : a Formula for a step function of time

events : a convenience function to generate sums of events

blocks : a convenience function to generate sums of blocks

convolve_functions : numerically convolve two functions of time

fourier_basis : a convenience function to generate a Fourier basis

"""

__docformat__ = 'restructuredtext'

import warnings
import numpy as np
import numpy.fft as FFT
from scipy.interpolate import interp1d
from string import join as sjoin

from nipy.algorithms.statistics.utils import combinations

from sympy import Function, DiracDelta, Symbol 
from sympy import sin as sympy_sin
from sympy import cos as sympy_cos
from sympy import pi as sympy_pi

import formula
from aliased import aliased_function, lambdify as alambdify, vectorize

t = formula.Term('t')

def fourier_basis(freq):
    """
    Formula for Fourier drift, consisting of sine and
    cosine waves of given frequencies.

    Inputs:
    =======

    freq : [float]
        Frequencies for the terms in the Fourier basis.

    Outputs:
    ========

    f : Formula

    Examples:
    =========
    
    >>> f=fourier_basis([1,2,3])
    >>> f.terms
    array([cos(2*pi*t), sin(2*pi*t), cos(4*pi*t), sin(4*pi*t), cos(6*pi*t),
           sin(6*pi*t)], dtype=object)
    >>> f.mean
    _b0*cos(2*pi*t) + _b1*sin(2*pi*t) + _b2*cos(4*pi*t) + _b3*sin(4*pi*t) + _b4*cos(6*pi*t) + _b5*sin(6*pi*t)
    >>>               
    """

    r = []
    for f in freq:
        r += [sympy_cos((2*sympy_pi*f*t)),
              sympy_sin((2*sympy_pi*f*t))]
    return formula.Formula(r)

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

    f : sympy expression 
        A Function of t.

    Examples:
    =========

    >>> s=linear_interp([0,4,5.],[2.,4,6], bounds_error=False)
    >>> tval = np.array([-0.1,0.1,3.9,4.1,5.1]).view(np.dtype([('t', np.float)]))
    >>> s.design(tval)
    array([(nan,), (2.0499999999999998,), (3.9500000000000002,),
           (4.1999999999999993,), (nan,)],
          dtype=[('interp0(t)', '<f8')])

    """
    kw['kind'] = 'linear'
    i = interp1d(times, values, **kw)

    if name is None:
        name = 'interp%d' % linear_interp.counter
        linear_interp.counter += 1

    s = aliased_function(name, i)
    return s(t)
linear_interp.counter = 0

# def event_factor(times_labels, f=DiracDelta):
#     """
#     Create a factor from a generator of
#     pairs of event times and labels.
#     """
#     val = {}
#     for time, label in times_labels:
#         val[label].setdefault(k, []).append(time)
#     regressors = []
#     for label in val.keys():
#         regressors.append(events(val[label]))
#     return Formula(regressors)

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
    >>> s.design(tval)
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

    s = aliased_function(name, anon)
    return s(t)
step_function.counter = 0

def events(times, amplitudes=None, f=DiracDelta, g=Symbol('a')):
    """
    Return a sum of functions
    based on a sequence of times.

    Parameters
    ----------

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

    Examples
    --------

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
    >>> b.design(tval)
    array([(0.0,), (1.0,), (0.0,), (1.0,)], 
          dtype=[('step0(t)', '<f8')])

    >>> b = blocks([[1,2],[3,4]], amplitudes=[3,5])
    >>> b.design(tval)
    array([(0.0,), (3.0,), (0.0,), (5.0,)], 
          dtype=[('step1(t)', '<f8')])

    >>> a = Symbol('a')
    >>> b = blocks([[1,2],[3,4]], amplitudes=[3,5], g=a+1)
    >>> b.design(tval)
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


def convolve_functions(fn1, fn2, interval, dt, padding_f=0.1, name=None):
    """
    Convolve fn1 with fn2.
    
    Parameters
    ----------
        fn1 : sympy expr
            An expression that is a function of t only.
        fn2 : sympy expr
            An expression that is a function of t only.
        interval : [float, float]
            The interval over which to convolve the two functions.
        dt : float
            Time step for discretization 
        padding_f : float
            Padding added to the left and right in the convolution.
        name : str
            Name of the convolved function in the resulting expression. 
            Defaults to one created by linear_interp.
    Returns
    -------
    f : sympy expr
            An expression that is a function of t only.

    >>> t = sympy.Symbol('t')
    >>> # This is a square wave on [0,1]
    >>> f1 = (t > 0) * (t < 1)
    >>> # The convolution of with itself is a triangular wave on [0,2], peaking at 1 with height 1
    >>> tri = convolve_functions(f1, f1, [0,2], 1.0e-03, name='conv')
    >>> print tri
    conv(t)
    >>> ftri = vectorize(tri)
    >>> x = np.linspace(0,2,11)
    >>> y = ftri(x)
    >>> # This is the resulting y-value (which seem to be numerically off by dt
    >>> y
    array([ -3.90255908e-16,   1.99000000e-01,   3.99000000e-01,
           5.99000000e-01,   7.99000000e-01,   9.99000000e-01,
           7.99000000e-01,   5.99000000e-01,   3.99000000e-01,
           1.99000000e-01,   6.74679706e-16])
    >>> 
    """

    max_interval, min_interval = max(interval), min(interval)
    ltime = max_interval - min_interval
    time = np.arange(min_interval, max_interval + padding_f * ltime, dt)

    f1 = vectorize(fn1)
    f2 = vectorize(fn2)
    _fn1 = np.array(f1(time))
    _fn2 = np.array(f2(time))

    _fft1 = FFT.rfft(_fn1)
    _fft2 = FFT.rfft(_fn2)

    value = FFT.irfft(_fft1 * _fft2) * dt
    _minshape = min(time.shape[0], value.shape[-1])
    time = time[0:_minshape]
    value = value[0:_minshape]

    return linear_interp(time + min_interval, value, bounds_error=False, name=name)

def make_design(event_spec, t, order=2, hrfs=[DiracDelta]):
    """
    Create a design matrix for a GLM analysis based
    on an event specification, evaluating
    it a sequence of time values. Each column
    in the design matrix will be convolved with each HRF in hrfs.

    Parameters:
    -----------

    event_spec : np.recarray
        A recarray having at least a field named 'time' signifying
        the event time, and all other fields will be treated as factors
        in an ANOVA-type model.

    t : np.ndarray
        An array of np.float values at which to evaluate
        the design. Common examples would be the acquisition
        times of an fMRI image.

    order : int
        The highest order interaction to be considered in
        constructing the contrast matrices.

    hrfs : seq
        A sequence of (symbolic) HRF that will be convolved
        with each event. If empty, glover is used.

    Outputs: 
    --------
    
    X : np.ndarray
        The design matrix with X.shape[0] == t.shape[0]. The number
        of columns will depend on the other fields of event_spec.

    contrasts : dict
        Dictionary of contrasts that is expected to be of interest
        from the event specification. For each interaction / effect
        up to a given order will be returned. Also, a contrast
        is generated for each interaction / effect for each HRF
        specified in hrfs.
    
    """

    fields = list(event_spec.dtype.names)
    if 'time' not in fields:
        raise ValueError('expecting a field called "time"')

    fields.pop(fields.index('time'))
    e_factors = [formula.Factor(n, np.unique(event_spec[n])) for n in fields]
    
    e_formula = np.product(e_factors)

    e_contrasts = {}
    for i in range(1, order+1):
        for comb in combinations(zip(fields, e_factors), i):
            names = [c[0] for c in comb]
            fs = [c[1].main_effect for c in comb]
            e_contrasts[sjoin(names, ':')] = np.product(fs).design(event_spec)

    e_contrasts['constant'] = formula.I.design(event_spec)

    # Design and contrasts in event space
    # TODO: make it so I don't have to call design twice here
    # to get both the contrasts and the e_X matrix as a recarray

    e_X = e_formula.design(event_spec)
    e_dtype = e_formula.dtype

    # Now construct the design in time space

    t_terms = []
    t_contrasts = {}
    for l, h in enumerate(hrfs):
        t_terms += [events(event_spec['time'], \
            amplitudes=e_X[n], f=h) for i, n in enumerate(e_dtype.names)]
        for n, c in e_contrasts.items():
            t_contrasts["%s_%d" % (n, l)] = formula.Formula([ \
                 events(event_spec['time'], amplitudes=c[nn], f=h) for i, nn in enumerate(c.dtype.names)])
    t_formula = formula.Formula(t_terms)
    
    tval = formula.make_recarray(t, ['t'])
    X_t, c_t = t_formula.design(tval, contrasts=t_contrasts)
    return X_t, c_t
