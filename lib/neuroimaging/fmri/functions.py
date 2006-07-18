"""
This module defines functions of time and tools to manipulate them.

The main class TimeFunction, is a function from (real) time to an arbitrary
number of (real) outputs.

These objects can be (coordinate-wised) multiplied, added, subtracted and
divided.

"""

import types

import numpy as N
from neuroimaging import traits

from scipy.sandbox.models.utils import recipr0, StepFunction

# Prototypical stimuli: "Event" (on/off) and "Stimuli" (step function)
# -Event inherits from Stimulus so most functionality is in Stimulus
# -changes are just in specifying parameters of self.fn

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

