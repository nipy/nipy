"""
This module defines functions of time and tools to manipulate them.

The main class TimeFunction, is a function from (real) time to an arbitrary
number of (real) outputs.

These objects can be (coordinate-wised) multiplied, added, subtracted and
divided.

"""

__docformat__ = 'restructuredtext'

import numpy as N

from scipy.sandbox.models.utils import recipr0
from scipy.sandbox.models.utils import StepFunction
from scipy.interpolate import interp1d

# Prototypical stimuli: "Event" (on/off) and "Stimuli" (step function)
# -Event inherits from Stimulus so most functionality is in Stimulus
# -changes are just in specifying parameters of self.fn

class TimeFunction(object):

    def __init__(self, fn, nout=1, slice=None, windowed=False, window=(0., 0.),
                 name=""):
        self.name = name
        self.windowed = windowed
        self.window = window
        self.fn = fn
        self.nout = nout
	self.slice = slice

    def __getitem__(self, j):

        def _f(time, obj=self):
            return obj(time)[int(j)]
        return TimeFunction(fn=_f)

    def __call__(self, time):
        columns = []

        if self.nout == 1:
            # fn is a single function with one output
            columns.append(self.fn(time))
        else:
            if isinstance(self.fn, (list, tuple)):
                # fn is a list of functions with one output
                for fn in self.fn:
                    columns.append(fn(time))
            else:
                # fn is a single function with a list of outputs
                columns = self.fn(time)

        if self.windowed:
            _window = N.greater(time, self.window[0]) * N.less_equal(time, self.window[1])
            columns = [column * _window for column in columns]
                
        if not self.slice:
            return N.squeeze(N.array(columns))
        else:
            return N.squeeze(N.array(columns[self.slice]))

    def _helper(self, other, f1, f2, f3):
        """
        All the operator overloads follow this same pattern
        doing slightly different things for f1, f2 and f3
        """
        if isinstance(other, TimeFunction):
            if other.nout == self.nout:
                _f = f1
            else:
                raise ValueError, 'number of outputs of regressors do not match'
        elif isinstance(other, (float, int)):
            _f = f2
        elif isinstance(other, (list, tuple, N.ndarray)):
            if isinstance(other, N.ndarray):
                if other.shape != (self.nout,):
                    raise 'shape does not much output, ' \
                          'expecting (%d,)' % self.nout
            elif len(other) != self.nout:
                raise 'length does not much output, expecting sequence of' \
                      'length %d' % self.nout
            _f = f3
        else:
            raise ValueError, 'unrecognized type'
        return TimeFunction(fn=_f, nout=self.nout)

    def __mul__(self, other):

        def f1(time, _self=self, _other=other):
            return N.squeeze(_self(time) * _other(time))

        def f2(time, _self=self, _other=other):
            return N.squeeze(_self(time) * _other)

        def f3(time, _self=self, _other=N.array(other)):
            v = _self(time)
            for i in range(_other.shape[0]):
                v[i] *= _other[i]
            return N.squeeze(v)

        return self._helper(other, f1, f2, f3)

    def __add__(self, other):
        def f1(time, _self=self, _other=other):
            v = _self(time) + _other(time)
            return N.squeeze(v)

        def f2(time, _self=self, _other=other):
            v = _self(time) + _other
            return N.squeeze(v)

        def f3(time, _self=self, _other=N.array(other)):
            v = _self(time)
            for i in range(_other.shape[0]):
                v[i] += _other[i]
            return N.squeeze(v)

        return self._helper(other, f1, f2, f3)


    def __sub__(self, other):
        def f1(time, _self=self, _other=other):
            v = _self(time) - _other(time)
            return N.squeeze(v)

        def f2(time, _self=self, _other=other):
            v = _self(time) - _other
            return N.squeeze(v)


        def f3(time, _self=self, _other=N.array(other)):
            v = _self(time)
            for i in range(_other.shape[0]):
                v[i] -= _other[i]
            return N.squeeze(v)

        return self._helper(other, f1, f2, f3)

    def __div__(self, other):
        def f1(time, _self=self, _other=other):
            return N.squeeze(_self(time) * recipr0(_other(time)))

        def f2(time, _self=self, _other=other):
            return N.squeeze(_self(time) * recipr0(_other))

        def f3(time, _self=self, _other=N.array(other)):
            v = _self(time) 
            for i in range(_other.shape[0]):
                v[i] *= recipr0(_other[i])
            return N.squeeze(v)
        
        return self._helper(other, f1, f2, f3)



class InterpolatedConfound(TimeFunction):

    def __init__(self, times=None, values=None, **keywords):

        if times is None:
            self.times = []
        else:
            self.times = times

        if values is None:
            self.values = []
        else:
            self.values = values

        if len(N.asarray(self.values).shape) == 1:
            self.f = interp1d(self.times, self.values, bounds_error=0)
            self.nout = 1
        else:
            self.f = []
            values = N.asarray(self.values)
            for i in range(values.shape[0]):
                f = interp1d(self.times, self.values[:, i], bounds_error=0)
                self.f.append(f)
            self.nout = values.shape[0]
            
        TimeFunction.__init__(self, self.f, nout=self.nout, **keywords)

    def __call__(self, time):
        columns = []

        if self.nout == 1:
            columns.append(self.f(time))
        else:
            if isinstance(self.f, (list, tuple)):
                for f in self.f:
                    columns.append(f(time))
            else:
                columns = self.f(time)

        if self.windowed:
            _window = N.greater(time, self.window[0]) * \
                      N.less_equal(time, self.window[1])
            columns = [column * _window for column in columns]
                
        return N.squeeze(N.array(columns))

class Stimulus(TimeFunction):

    def __init__(self, fn, times=None, values=None, **keywords):
        TimeFunction.__init__(self, fn, **keywords)
        if times is None:
            self.times = []
        else:
            self.times = times

        if values is None:
            self.values = []
        else:
            self.values = values

class PeriodicStimulus(Stimulus):

    def __init__(self, n=1, start=0.0, duration=3.0, step=6.0, height=1.0,
                 **keywords):
        self.n = n
        self.start = start
        self.duration = duration
        self.step = step
        self.height = height

        times = [-1.0e-07]
        values = [0.]

        for i in range(self.n):
            times = times + [self.step*i + self.start,
                             self.step*i + self.start + self.duration]
            values = values + [self.height, 0.]
        Stimulus.__init__(self, times=times, values=values, **keywords)

class Events(Stimulus):

    def __init__(self, **keywords):
        Stimulus.__init__(self, None, **keywords)

    def append(self, start, duration, height=1.0):
        """
        Append a square wave to an Event. No checking is made
        to ensure that there is no overlap with previously defined
        intervals -- the assumption is that this new interval
        has empty intersection with all other previously defined intervals.
        """
        
        if self.times is None:
            self.times = []
            self.values = []
            self.fn = lambda x: 0.

        times = N.array(list(self.times) + [start, start + duration])
        asort = N.argsort(times)
        values = N.array(list(self.values) + [height, 0.])

        self.times = N.take(times, asort)
        self.values = N.take(values, asort)

        self.fn = StepFunction(self.times, self.values, sorted=True)

class DeltaFunction(TimeFunction):

    """
    A square wave approximate delta function returning
    1/dt in interval [start, start+dt).
    """

    def __init__(self, start=0.0, dt=0.02):
        """
        @param start: Beginning of delta function approximation.
        @type start: C{float}
        @param dt: Width of delta function approximation.
        @type dt: C{float}
        """
        TimeFunction.__init__(self, None)
        self.start = start
        self.dt = dt

    def __call__(self, time):
        return N.greater_equal(time, self.start) * \
               N.less(time, self.start + self.dt) / self.dt

class SplineConfound(TimeFunction):

    """
    A natural spline confound with df degrees of freedom.
    """
    
    def __init__(self, df=4, knots=None, **keywords):

        TimeFunction.__init__(self, None, **keywords)
        self.df = df
        if knots is None:
            self.knots = []
        else:
            self.knots = knots
        tmax = self.window[1]
        tmin = self.window[0]
        trange = tmax - tmin

        self.fn = []

        def getpoly(j):
            def _poly(time=None):
                return time**j
            return _poly

        for i in range(min(self.df, 4)):
            self.fn.append(getpoly(i))

        if self.df >= 4 and not self.knots:
            self.knots = list(trange * N.arange(1, self.df - 2) / (self.df - 3.0) + tmin)
        self.knots[-1] = N.inf 

        def _getspline(a, b):
            def _spline(time):
                return N.power(time - a, 3.0) * N.greater(time, a) * \
                       N.less_equal(time, b)
            return _spline

        for i in range(len(self.knots) - 1):
            self.fn.append(_getspline(self.knots[i], self.knots[i+1]))

        self.nout = self.df
