"""
This module defines some convenience functions of time.

Events: subclass of Stimulus to which events can be appended

InterpolatedConfound: based on a sequence of [times, values], return
                      a linearly interpolated confound

"""

__docformat__ = 'restructuredtext'

import numpy as np

from neuroimaging.fixes.scipy.stats.models.utils import recipr0
from neuroimaging.fixes.scipy.stats.models.utils import StepFunction
from scipy.interpolate import interp1d

# Prototypical stimuli: "Event" (on/off) and "Stimuli" (step function)
# -Event inherits from Stimulus so most functionality is in Stimulus
# -changes are just in specifying parameters of self.fn

from sympy import Symbol, Function, lambdify, DeferredVector, DiracDelta
#from sympy.decorator import deprecated

vector_t = DeferredVector('vt')
t = Symbol('t')

def generic_function(name, function):
    """
    Create a named, generic function of time
    for functions that do not have efficient
    sympy representations.

    Inputs:
    =======

    name : str
        name of function

    function : callable
        function of one variable

    Outputs:
    ========

    f(t)

    Examples:
    =========

    >>> from scipy.interpolate import interp1d
    >>> s = interp1d([0,3,5.],[2,4,6.], kind='linear')
    >>> g = generic_function('f', s)
    >>> gv = vectorize_generic(g)
    >>> print g
    f(t)
    >>> gv([0,3,5])
    array([ 2.,  4.,  6.])
    >>>                                 

    """
    new = Function(name, dummy=True)(t)
    new._name = name
    new._generic_callable = function
    return new

def vectorize_generic(f):
    """
    Take a sympy expression that contains the symbol 't'
    and return a lambda with a vectorized time.
    """
    return lambdify(vector_t, f.subs(t, vector_t), {f._name:f._generic_callable})        

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

    f : sympy.Add
        A sympy expression for a step function.

    Examples:
    =========

    >>> f = step_function([0,4,5],[1,-1,3])
    >>> print f
    -2*(4 <= t) + 4*(5 <= t) + (0 <= t)
    >>>

    """
    times = np.asarray(times)
    values = np.asarray(values)
    d = values[1:] - values[:-1]
    f = (t < times[0]) * fill + (t >= times[0]) * values[0]
    for i in range(d.shape[0]):
        f = f + (t >= times[i+1]) * d[i]
    return f

def events(generator):
    """
    Return a sum of DiracDelta functions
    based on a generator of times.

    Inputs:
    =======

    generator : [float]

    Examples:
    =========

    >>> events([3,6,9])
    DiracDelta(-9 + t) + DiracDelta(-6 + t) + DiracDelta(-3 + t)

    """
    e = 0
    for _t in generator:
        e = e + DiracDelta(t-_t)
    return e

def events_and_amplitudes(generator):
    """
    Return a sum of DiracDelta functions
    based on a generator of times.

    Inputs:
    =======

    generator : [(float, float)]

    Examples:
    =========

    >>> events_and_amplitudes(np.array([[2,3],[1,6],[-1,9]]))
    -DiracDelta(-9 + t) + 2*DiracDelta(-3 + t) + DiracDelta(-6 + t)

    >>> from sympy import Symbol
    >>> b = [Symbol('b%d' % i, dummy=True) for i in range(3)]
    >>> a = Symbol('a')
    >>> p = b[0] + b[1]*a + b[2]*a**2
    >>> events_and_amplitudes(np.array([[p.subs(a,2),3],[p.subs(a,1),6],[p.subs(a,-1),9]]))
    (2*_b1 + 4*_b2 + _b0)*DiracDelta(-3 + t) + (-_b1 + _b0 + _b2)*DiracDelta(-9 + t) + (_b0 + _b1 + _b2)*DiracDelta(-6 + t)
    >>>                              
    """
    e = 0
    for a, _t in generator:
        e = e + a * DiracDelta(t-_t)
    return e

def linear_interp(times, values, fill=0):
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

    f : sympy.Add
        A sympy expression for a linear interpolant.

    Examples:
    =========

    >>> f = linear_interp([0,1,2],[0,2,0])
    >>> f
    (2 - 2*t)*(1 <= t)*(t <= 2) + 2*t*(0 <= t)*(t <= 1)

    """
    a = np.asarray(values[:-1])
    b = np.asarray(values[1:])
    f = values[0]
    for i in range(a.shape[0]):
        g = ((t  - times[i]) * (t >= times[i]) * (t <= times[i+1]) /
             (times[i+1] - times[i]) * (values[i+1] - values[i]))
        g = g + (values[i+1] - values[i]) * (t > times[i+1])
        f = f + g
    return f

