"""
This module defines some convenience functions of time.

Stimulus: a class to implement square-wave protocol based on
          a pair of sequences [times,values]

PeriodicStimulus: periodic Stimulus

Events: subclass of Stimulus to which events can be appended

DeltaFunction: an approximate delta function

SplineConfound: generate natural cubic splines with given knots

InterpolatedConfound: based on a sequence of [times, values], return
                      a linearly interpolated confound

"""

__docformat__ = 'restructuredtext'

import numpy as np

from nipy.fixes.scipy.stats.models.utils import recipr0
from nipy.fixes.scipy.stats.models.utils import StepFunction
from scipy.interpolate import interp1d

# Prototypical stimuli: "Event" (on/off) and "Stimuli" (step function)
# -Event inherits from Stimulus so most functionality is in Stimulus
# -changes are just in specifying parameters of self.fn

def window(f, r):
    """
    Decorator to window a function between r[0] and r[1] (inclusive)
    """
    def h(f):
        def g(x):
            return np.greater_equal(x, r[0]) * np.less_equal(x, r[1]) * f(x)
        g.window = r
        return g
    return h

class Stimulus:
    """
    TODO
    """

    def __init__(self, name='stimulus', times=None, values=None):
        """
        :Parameters:
            `fn` : TODO
                TODO
            `times` : TODO
                TODO
            `values` : TODO
                TODO
        """
        if times is None:
            self.times = []
        else:
            self.times = times

        if values is None:
            self.values = []
        else:
            self.values = values

        if self.times:
            a = np.argsort(self.times)
            self.values = self.values[a]
            self.times = self.times[a]

        self.fn = StepFunction(self.times, self.values, sorted=True)

    def __call__(self, t):
        """
        Right continuous stimulus.
        """
        return self.fn(t)

class PeriodicStimulus(Stimulus):
    """
    TODO
    """

    def __init__(self, n=1, start=0.0, duration=3.0, step=6.0, height=1.0,
                 name='periodic stimulus'):

        """
        :Parameters:
            `n` : int
                TODO
            `start` : float
                TODO
            `duration` : float
                TODO
            `step` : float
                TODO
            `height` : float
                TODO
        """
        self.n = n
        self.start = start
        self.duration = duration
        self.step = step
        self.height = height

        times = [start-1.0e-07]
        values = [0.]

        for i in range(self.n):
            times = times + [self.step*i + self.start,
                             self.step*i + self.start + self.duration]
            values = values + [self.height, 0.]
        Stimulus.__init__(self, times=times, values=values, name=name)

class Events(Stimulus):
    """
    TODO
    """

    def append(self, start, duration, height=1.0):
        """
        Append a square wave to an Event. No checking is made
        to ensure that there is no overlap with previously defined
        intervals -- the assumption is that this new interval
        has empty intersection with all other previously defined intervals.
        
        :Parameters:
            `start` : TODO
                TODO
            `duration` : TODO
                TODO
            `height` : float
                TODO
                
        :Returns: ``None``
        """
        
        if self.times is None:
            self.times = []
            self.values = []
            self.fn = lambda x: 0.

        times = np.array(list(self.times) + [start, start + duration])
        asort = np.argsort(times)
        values = np.array(list(self.values) + [height, 0.])

        self.times = times[asort]
        self.values = values[asort]

        self.fn = StepFunction(self.times, self.values, sorted=True)

class DeltaFunction:

    """
    A square wave approximate delta function returning
    1/dt in interval [start, start+dt).
    """

    def __init__(self, start=0.0, dt=0.02):
        """
        :Parameters:
            `start` : float
                Beginning of delta function approximation.
            `dt` : float
                Width of delta function approximation.
        """
        self.start = start
        self.dt = dt

    def __call__(self, time):
        """
        :Parameters:
            `time` : TODO
                TODO
        
        :Returns: TODO
        """
        return np.greater_equal(time, self.start) * \
               np.less(time, self.start + self.dt) / self.dt

class SplineConfound:

    """
    A natural spline confound with df degrees of freedom.
    """
    
    def __init__(self, df=4, knots=None, window=[0,1]):
        """
        :Parameters:
            `df` : int
                TODO
            `knots` : TODO
                TODO
        """

        self.df = df
        if knots is None:
            self.knots = []
        else:
            self.knots = knots
        tmax = window[1]
        tmin = window[0]
        trange = tmax - tmin

        self.fn = []

        def getpoly(j):
            def _poly(time=None):
                return time**j
            return _poly

        for i in range(min(self.df, 4)):
            self.fn.append(getpoly(i))

        # FIXME: This trange is calculated different than above!  Do
        # this calculation correctly and do it once.
        trange = window[0] - window[1]
        tmin = window[0]
        
        if self.df >= 4 and not self.knots:
            self.knots = list(trange * np.arange(1, self.df - 2) / (self.df - 3.0) + tmin)
        self.knots[-1] = np.inf 

        def _getspline(a, b):
            def _spline(time):
                return np.power(time - a, 3.0) * np.greater(time, a) 
            return _spline

        for i in range(len(self.knots) - 1):
            self.fn.append(_getspline(self.knots[i], self.knots[i+1]))

        self.nout = self.df

    def __call__(self, t):
        if type(self.fn) in [type([]), type(())]:
            return np.asarray([f(t) for f in self.fn])
        else:
            return self.fn(t)

class InterpolatedConfound:

    def __init__(self, times=None, values=None, name='confound'):
        """
        :Parameters:
        `times` : TODO
        TODO
        `values` : TODO
        TODO
        `name` : TODO
        TODO

        """
        if times is None:
            self.times = []
        else:
            self.times = times
            
        if values is None:
            self.values = []
        else:
            self.values = values

        if len(np.asarray(self.values).shape) == 1:
            self.f = interp1d(self.times, self.values, bounds_error=0)
        else:
            self.f = []
            values = np.asarray(self.values)
            for i in range(values.shape[0]):
                f = interp1d(self.times, self.values[i, :], bounds_error=0)
                self.f.append(f)

    def __call__(self, time):
        """
        :Parameters:
        `time` : TODO
        TODO
        
        :Returns: TODO
        """
        
        if isinstance(self.f, (list, tuple)):
            columns = []
            for f in self.f:
                columns.append(f(time))
        else:
            columns = self.f(time)
            
        return np.squeeze(np.asarray(columns))



