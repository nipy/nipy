"""
This module defines functions of time and tools to manipulate them.

The main class TimeFunction, is a function from (real) time to an arbitrary
number of (real) outputs.

These objects can be (coordinate-wised) multiplied, added, subtracted and
divided.

"""

import numpy as N
from utils import StepFunction
import fpformat, types

from neuroimaging.statistics.utils import recipr0

# Use scipy's interpolator

from scipy.interpolate import interp1d

# Prototypical stimuli: "Event" (on/off) and "Stimuli" (step function)
# -Event inherits from Stimulus so most functionality is in Stimulus
# -changes are just in specifying parameters of self.fn

import enthought.traits as traits
times = N.arange(0,50,0.1)

class TimeFunction(traits.HasTraits):

    nout = traits.Int(1)
    fn = traits.Any()

    windowed = traits.false
    window = traits.List([0.,0.])

    def __getitem__(self, j):

        def _f(time=None, obj=self, **extra):
            return obj(time=time, **extra)[int(j)]
        return TimeFunction(fn=_f)

    def __call__(self, time=None, **extra):

        lt = len(N.array(time).shape)

        columns = []

        if self.nout == 1:
            columns.append(self.fn(time=time))
        else:
            if type(self.fn) in [types.ListType, types.TupleType]:
                for fn in self.fn:
                    columns.append(fn(time=time))
            else:
                columns = self.fn(time=time)

        if self.windowed:
            _window = N.greater(time, self.window[0]) * N.less_equal(time, self.window[1])
            columns = [column * _window for column in columns]
                
        return N.squeeze(N.array(columns))

    def __mul__(self, other):
        if isinstance(other, TimeFunction):
            if other.nout == self.nout:
                def _f(time=None, _self=self, _other=other, **extra):
                    return N.squeeze(_self(time=time, **extra) * _other(time=time, **extra))
            else:
                raise ValueError, 'number of outputs of regressors do not match'
        elif type(other) in [types.FloatType, types.IntType]:
            def _f(time=None, _self=self, _other=other, **extra):
                return N.squeeze(_self(time=time, **extra) * other)
        elif type(other) in [types.ListType, types.TupleType, N.ndarray]:
            if type(other) is N.ndarray:
                if other.shape != (self.nout,):
                    raise 'shape does not much output, expecting (%d,)' % self.nout
            elif len(other) != self.nout:
                    raise 'length does not much output, expecting sequence of length %d' % self.nout
            def _f(time=None, _self=self, _other=N.array(other), **extra):
                v = _self(time=time, **extra)
                for i in range(_other.shape[0]):
                    v[i] *= _other[i]
                return N.squeeze(v)
        else:
            raise ValueError, 'unrecognized type'
        return TimeFunction(fn=_f, nout=self.nout)

    def __add__(self, other):
        if isinstance(other, TimeFunction):
            if other.nout == self.nout:
                def _f(time=None, _self=self, _other=other, **extra):
                    v = _self(time=time, **extra) + _other(time=time, **extra)
                    return N.squeeze(v)
            else:
                raise ValueError, 'number of outputs of regressors do not match'
        elif type(other) in [types.FloatType, types.IntType]:
            def _f(time=None, _self=self, _other=other, **extra):
                v = _self(time=time, **extra) + other
                return N.squeeze(v)
        elif type(other) in [types.ListType, types.TupleType, N.ndarray]:
            if type(other) is N.ndarray:
                if other.shape != (self.nout,):
                    raise 'shape does not much output, expecting (%d,)' % self.nout
            elif len(other) != self.nout:
                    raise 'length does not much output, expecting sequence of length %d' % self.nout
            def _f(time=None, _self=self, _other=N.array(other), **extra):
                v = _self(time=time, **extra)
                for i in range(_other.shape[0]):
                    v[i] += _other[i]
                return N.squeeze(v)
        else:
            raise ValueError, 'unrecognized type'
        return TimeFunction(fn=_f, nout=self.nout)

    def __sub__(self, other):
        if isinstance(other, TimeFunction):
            if other.nout == self.nout:
                def _f(time=None, _self=self, _other=other, **extra):
                    v = _self(time=time, **extra) - _other(time=time, **extra)
                    return N.squeeze(v)
            else:
                raise ValueError, 'number of outputs of regressors do not match'
        elif type(other) in [types.FloatType, types.IntType]:
            def _f(time=None, _self=self, _other=other, **extra):
                v = _self(time=time, **extra) - other
                return N.squeeze(v)
        elif type(other) in [types.ListType, types.TupleType, N.ndarray]:
            if type(other) is N.ndarray:
                if other.shape != (self.nout,):
                    raise 'shape does not much output, expecting (%d,)' % self.nout
            elif len(other) != self.nout:
                    raise 'length does not much output, expecting sequence of length %d' % self.nout
            def _f(time=None, _self=self, _other=N.array(other), **extra):
                v = _self(time=time, **extra)
                for i in range(_other.shape[0]):
                    v[i] -= _other[i]
                return N.squeeze(v)
        else:
            raise ValueError, 'unrecognized type'
        return TimeFunction(fn=_f, nout=self.nout)

    def __div__(self, other):
        if isinstance(other, TimeFunction):
            if other.nout == self.nout:
                def _f(time=None, _self=self, _other=other, **extra):
                    return N.squeeze(_self(time=time, **extra) * recipr0(_other(time=time, **extra)))
            else:
                raise ValueError, 'number of outputs of regressors do not match'
        elif type(other) in [types.FloatType, types.IntType]:
            def _f(time=None, _self=self, _other=other, **extra):
                return N.squeeze(_self(time=time, **extra) * recipr0(other))
        elif type(other) in [types.ListType, types.TupleType, N.ndarray]:
            if type(other) is N.ndarray:
                if other.shape != (self.nout,):
                    raise 'shape does not much output, expecting (%d,)' % self.nout
            elif len(other) != self.nout:
                    raise 'length does not much output, expecting sequence of length %d' % self.nout
            def _f(time=None, _self=self, _other=N.array(other), **extra):
                v = _self(time=time, **extra) 
                for i in range(_other.shape[0]):
                    v[i] *= recipr0(_other[i])
                return N.squeeze(v)
        else:
            raise ValueError, 'unrecognized type'
        return TimeFunction(fn=_f, nout=self.nout)

class Stimulus(TimeFunction):

    times = traits.Any()
    values = traits.Any()

    def __init__(self, **keywords):
        TimeFunction.__init__(self, **keywords)

class PeriodicStimulus(Stimulus):
    n = traits.Int(1)
    start = traits.Float(0.)
    duration = traits.Float(3.0)
    step = traits.Float(6.0) # gap between end of event and next one
    height = traits.Float(1.0)

    def __init__(self, **keywords):

        traits.HasTraits.__init__(self, **keywords)
        times = [-1.0e-07]
        values = [0.]

        for i in range(self.n):
            times = times + [self.step*i + self.start, self.step*i + self.start + self.duration]
            values = values + [self.height, 0.]
        Stimulus.__init__(self, times=times, values=values, **keywords)

class Events(Stimulus):

    def __init__(self, **keywords):
        Stimulus.__init__(self, **keywords)

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

class InterpolatedConfound(TimeFunction):

    times = traits.Any()
    values = traits.Any()

    def __init__(self, **keywords):
        TimeFunction.__init__(self, **keywords)
        if len(N.asarray(self.values).shape) == 1:
            self.fn = interp1d(self.times, self.values, bounds_error=0)
            self.nout = 1
        else:
            self.fn = []
            values = N.asarray(self.values)
            for i in range(values.shape[0]):
                fn = interp1d(self.times, self.values[:,i], bounds_error=0)
                self.fn.append(fn)
            self.nout = values.shape[0]
            
    def __call__(self, time=None, **extra):

        lt = len(N.array(time).shape)

        columns = []

        if self.nout == 1:
            columns.append(self.fn(time))
        else:
            if type(self.fn) in [types.ListType, types.TupleType]:
                for fn in self.fn:
                    columns.append(fn(time))
            else:
                columns = self.fn(time)

        if self.windowed:
            _window = N.greater(time, self.window[0]) * N.less_equal(time, self.window[1])
            columns = [column * _window for column in columns]
                
        return N.squeeze(N.array(columns))


class FunctionConfound(TimeFunction):

    def __init__(self, fn=[], **keywords):
        '''
        Argument "fn" should be a sequence of functions describing the regressor.
        '''
        TimeFunction.__init__(self, **keywords)
        self.nout = len(fn)
        if len(fn) == 1:
            self.fn = fn[0]
        else:
            self.fn = fn

class SplineConfound(FunctionConfound):

    df = traits.Int(4)
    knots = traits.List()

    def __init__(self, **keywords):
        '''
        Basic spline trend confound.
        '''

        TimeFunction.__init__(self, **keywords)
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
            def _spline(time=None):
                return time**3 * N.greater(time, a) * N.less_equal(time, b)
            return _spline

        for i in range(len(self.knots) - 1):
            self.fn.append(_getspline(self.knots[i], self.knots[i+1]))

        self.nout = self.df
        
