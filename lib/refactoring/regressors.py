from numpy import *
import hrf, filters
from miscutils import StepFunction

# Use scipy's interpolator

from scipy.interpolate import interp1d
interpolant = interp1d


# Prototypical stimuli: "Event" (on/off) and "Stimuli" (step function)
# -Event inherits from Stimulus so most functionality is in Stimulus
# -changes are just in specifying parameters of self.fn

import enthought.traits as TR
times = arange(0,50,0.1)

class Regressor(TR.HasTraits):

    index =TR.Int(0)
    nout = TR.Int(1)
    tstat = TR.true
    fstat = TR.false
    effect = TR.true
    sd = TR.true
    name = TR.Str()
    fn = TR.Any()
    IRF = TR.Any()

    windowed = TR.false
    window = TR.List([0.,0.])

    def __call__(self, times=times, **extra):

        columns = []
        if self.IRF is not None:

            if not hasattr(self, 'base_fn') and self.IRF is not None:
                interval = (0, max(self.times) + filters.Filter.tpad)
                time = arange(interval[0], interval[1], filters.Filter.dt)
                if self.nout != 1:
                    raise ValueError, 'expecting 1D function for convolution.'

                self.base_fn = self.fn
                self.fn = self.IRF.convolve(self.base_fn, interval, dt=filters.Filter.dt)
            self.nout = self.IRF.n
        
        if self.nout == 1:
            columns.append(self.fn(times))
        else:
            for fn in self.fn:
                columns.append(fn(times))

        if self.windowed:
            _window = greater(times, self.window[0]) * less_equal(times, self.window[1])
            columns = [column * _window for column in columns]
                
        return columns

    def get_name(self, n=0, extra='basis'):
        if self.nout == 1:
            return self.name
        else:
            return self.name + '_%s%d' % (extra, n)

##     def setup_contrasts(self, output_base=''):

##         # T contrasts

##         if self.sd or self.effect or self.tstat:
##             self.Tcontrasts = [Tcontrast(output_base + '_' + self.get_name(), column=self.index, sd=self.sd, eff=self.effect, t=self.tstat)]
##             self.Tcontrasts[0].shortname = self.get_name()
##         else:
##             self.Tcontrasts = []
##             for k in range(self.nout):
##                 if self.sd or self.effect or self.tstat:
##                     curcontrast = Tcontrast('%s_%s' % (output_base, self.get_name(n=k)), column=self.index+k, sd=self.sd, eff=self.effect, t=self.tstat)
##                     curcontrast.shortname = self.get_name(n=k)
##                     self.Tcontrasts.append(curcontrast)

##         if self.fstat:
##             self.Fcontrast = Fcontrast(output_base + '_' + self.name + '_F', columns=range(self.index, self.index.self.nout))


class Stimulus(Regressor):

    times = TR.Any()
    values = TR.Any()

    def __init__(self, IRF=None, **keywords):
        Regressor.__init__(self, **keywords)
        self.IRF = IRF

class PeriodicStimulus(Stimulus):
    n = TR.Int(1)
    start = TR.Float(0.)
    duration = TR.Float(3.0)
    step = TR.Float(6.0) # gap between end of event and next one
    height = TR.Float(1.0)

    def __init__(self, IRF=None, **keywords):

        TR.HasTraits.__init__(self, **keywords)
        times = [-1.0e-07]
        values = [0.]

        for i in range(self.n):
            times = times + [self.step*i + self.start, self.step*i + self.start + self.duration]
            values = values + [self.height, 0.]
        Stimulus.__init__(self, times=times, values=values, IRF=IRF, **keywords)

class Events(Stimulus):

    def __init__(self, IRF=None, **keywords):
        Stimulus.__init__(self, IRF=IRF, **keywords)

    def append(self, start, duration, height=1.0):
        if self.times is None:
            self.times = [-inf,+inf]
            self.values = [0, 0]
            self.fn = StepFunction(self.times, self.values)

        times = self.times + [start, start + duration]
        times.sort()
        self.times = times

        newtimes = [-inf, start, start + duration]
        newvalues = [0, height, 0]

        fn = StepFunction(newtimes, newvalues, sorted=True)
        y = fn(self.times)

        if self.fn is not None:
            self.values = y + self.fn(self.times)
        else:
            self.values = y

        self.fn = StepFunction(self.times, self.values)

class InterpolatedConfound(Regressor):

    times = TR.Any()
    values = TR.Any()

    def __init__(self, **keywords):
        Regressor.__init__(self, **keywords)
        if len(array(self.values).shape) == 1:
            self.fn = interpolant(self.times, self.values)
        else:
            self.fn = []
            values = array(self.values)
            for i in range(values.shape[0]):
                self.fn.append(self.times, values[:,i])

class FunctionConfound(Regressor):

    def __init__(self, fn=[], **keywords):
        '''
        Argument "fn" should be a sequence of functions describing the regressor.
        '''
        Regressor.__init__(self, **keywords)
        self.nout = len(fn)
        if len(fn) == 1:
            self.fn = fn[0]
        else:
            self.fn = fn

class SplineConfound(FunctionConfound):

    df = TR.Int(4)
    knots = TR.List()
    pad = TR.Float(5.0)

    def __init__(self, **keywords):
        '''
        Basic spline trend confound.
        '''

        Regressor.__init__(self, **keywords)
        tmax = self.window[1]
        tmin = self.window[0]
        trange = tmax - tmin

        self.fn = []

        def getpoly(j):
            def _poly(x):
                return x**j
            return _poly

        for i in range(min(self.df, 4)):
            self.fn.append(getpoly(i))

        if self.df >= 4 and not self.knots:
            self.knots = list(trange * arange(1, self.df - 2) / (self.df - 3.0) + tmin)
        self.knots[-1] = tmax + self.pad # buffer to fix "0 problem" at the end of the spline with slicetimes, will have to be thought out for concatenating runs

        def getspline(a, b):
            def _spline(x):
                return x**3 * greater(x, a) * less_equal(x, b)
            return _spline

        for i in range(len(self.knots) - 1):
            self.fn.append(getspline(self.knots[i], self.knots[i+1]))

        self.nout = self.df
        
