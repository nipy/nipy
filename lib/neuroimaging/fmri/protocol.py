import csv, types, copy
from enthought import traits
import numpy as N

from neuroimaging.fmri.functions import TimeFunction, StepFunction
from scipy.sandbox.models.formula import Factor, Quantitative, Formula, Term
from scipy.interpolate import interp1d

namespace = {}
downtime = 'None/downtime'

class ExperimentalRegressor(traits.HasTraits):

    convolved = traits.false # a toggle to determine whether we
                             # want to think of the factor as convolved or not
                             # i.e. for plotting

    def _convolved_changed(self):
        if not hasattr(self, '_nameunconv'):
            self._nameunconv = copy.copy(self.name)
        if self.convolved:
            self.name = self._convolved.name
        else:
            self.name = self._nameunconv

    def __add__(self, other):
        other = ExperimentalFormula(other)
        return other + self

    def __mul__(self, other):
        other = ExperimentalFormula(other)
        return other * self

    def names(self):
        if self.convolved:
            return self._convolved.names()
        else:
            if hasattr(self, '_nameunconv'):
                return self._nameunconv
            else:
                return Term.names(self)

    def convolve(self, IRF):

        self.IRF = IRF
        self.convolved = False

        func = IRF.convolve(self)

        def _f(time=None, func=tuple(func), **keywords):
            v = []
            for _func in func:
                try:
                    v.append(_func(time, **keywords))
                except:
                    for __func in _func:
                        v.append(__func(time, **keywords))
            return N.array(v)
        
        name = []
        for hrfname in IRF.names:
            for termname in self.names():
                name.append('(%s%%%s)' % (hrfname, termname))
 
        self._convolved = ExperimentalQuantitative(name, _f, termname='(HRF%%%s)' % self.termname)
        self.convolved = True

        return self

    def astimefn(self, namespace=namespace,
                 index=None):
        """
        Return a TimeFunction object that can be added, subtracted, etc.

        The values of the time function are determined by a linear
        interpolator based on self when astimefn is called.

        """

        nout = len(self.names())
        if index is not None:
            nout = 1

        def _f(time, index=index, namespace=namespace):
            v = self(time=time, namespace=namespace)
            if index is not None:
                v = v[index]
            return N.squeeze(v) * 1.

        v = TimeFunction(fn=_f, nout=nout)
        return v

class ExperimentalQuantitative(ExperimentalRegressor, Quantitative):
    """
    Generate a regressor that is a function of time
    based on a function fn.
    """

    def __init__(self, name, fn, termname=None, **keywords):

        ExperimentalRegressor.__init__(self, **keywords)
        self.fn = fn
        self.name = name
        if termname is None:
            termname = name
        namespace[termname] = self
            
        test = N.array(self.fn(time=N.array([4.0,5.0,6])))
        if test.ndim > 1:
            n = test.shape[0]
        else:
            n = 1
        if n > 1:
            if type(name) in [type([]), type(())]:
                names = name
            else:
                names = ['(%s:%d)' % (name, i) for i in range(n)]
        else:
            names = name

        Quantitative.__init__(self, names, func=fn, termname=termname, **keywords)
        
    def __call__(self, time=None, namespace=namespace, **keywords):
        if not self.convolved:
            return self.fn(time=time, **keywords)
        else:
            if hasattr(self, '_convolved'):
                return self._convolved(time=time, namespace=namespace, **keywords)
            else:
                raise ValueError, 'term %s has not been convolved with an HRF' % self.name
            
class Time(ExperimentalQuantitative):
    
    def __call__(self, time=None, **ignored):
        return time

    def __pow__(self, e):
        try:
            e = float(e)
        except:
            raise ValueError, 'only float exponents allowed'
        def _f(time=None, **ignored):
            return N.power(time, e)
        return ExperimentalQuantitative('time^%0.2f' % e, _f)
    
def _time(time=None): return time
Time = ExperimentalQuantitative('time', _time)

class ExperimentalStepFunction(ExperimentalQuantitative):
    """
    This returns a
    step function from an iterator returning tuples

    (start, stop, height)

    with height defaulting to 1 if not present.

    """

    def __init__(self, name, iterator, **keywords):

        fn = self.fromiterator(iterator)
        ExperimentalQuantitative.__init__(self, name, fn, **keywords)
  
    def fromiterator(self, iterator, delimiter=','):
        """
        Determine an ExperimentalStepFunction from an iterator
        which returns rows of the form:

        (start, stop, height)

        Here, Type is a hashable object and Start and Stop are floats.
        The fourth being an optional
        float for the height during the interval [Start,Stop] which
        defaults to 1.
        """

        if type(iterator) is types.StringType:
            iterator = csv.reader(file(iterator), delimiter=delimiter)
        elif type(iterator) is types.FileType:
            iterator = csv.reader(iterator, delimiter=delimiter)

        self.events = Events(name=self.name)

        for row in iterator:
            try:
                start, end, height = row
            except:
                start, end = row
                height = 1.0
                pass

            self.events[eventtype].append(float(start), float(end)-float(start), height=float(height))

        return self.events

class ExperimentalFactor(ExperimentalRegressor, Factor):
    """
    Return a factor that is a function of experimental time based on
    an iterator. If the delta attribute is False, it is assumed that
    the iterator returns rows of the form:

    type, start, stop

    Here, type is a hashable object and start and stop are floats.

    If delta is True, then the events are assumed to be delta functions
    and the rows are assumed to be of the form:

    type, start

    where the events are (square wave) approximations
    of a delta function, non zero on [start, start+dt). 

    """
    
    delta = traits.Trait(True, desc='Are the events delta functions?')
    dt = traits.Trait(0.02, desc='Width of the delta functions.')

    def __init__(self, name, iterator, **keywords):
        ExperimentalRegressor.__init__(self, **keywords)
        self.fromiterator(iterator)
        keys = self.events.keys() + [downtime]
        Factor.__init__(self, name, keys)
        self._event_keys = self.events.keys()
        self._event_keys.sort()
        
    def main_effect(self):
        """
        Return the 'main effect' for an ExperimentalFactor.
        """

        _c = self.convolved
        self.convolved = False
        f = self.astimefn()
        self.convolved = _c
        return ExperimentalQuantitative('%s:maineffect' % self.termname, f)

    def __getitem__(self, key):

        if self.events.has_key(key) not in self.events.keys():
            l = self.events.keys()
            j = l.index(key)
        else:
            raise ValueError, 'key not found'            

        def func(namespace=namespace, time=None, j=j,
                obj=self, 
                **extra):
            _c = obj.convolved
            obj.convolved = False
            v = obj(time=time, namespace=namespace)[j]
            obj.convolved = _c
            return [N.squeeze(v) * 1.]

        name = '%s[%s]' % (self.termname, `key`)
        return ExperimentalQuantitative(name, func)

    def __call__(self, time=None, namespace=None, includedown=False, convolved=None, **keywords):
        if convolved is not None:
            __convolved, self.convolved = self.convolved, convolved
        else:
            __convolved = self.convolved

        if not self.convolved:
            value = []
            keys = self._event_keys
            for level in keys:
                value.append(N.squeeze(self.events[level](time,
                                                          namespace=namespace,
                                                          **keywords)))
            if includedown:
                s = N.add.reduce(value)

                keys = keys + [downtime]
                which = N.argmax(value, axis=0)
                which = N.where(s, which, keys.index(downtime))
                value = Factor.__call__(self, namespace={self.termname:[keys[w] for w in which]})
            else:
                value = N.asarray(value)
        else:
            if hasattr(self, '_convolved'):
                value = self._convolved(time=time, namespace=namespace, **keywords)
            else:
                raise ValueError, 'no IRF defined for factor %s' % self.name
        self.convolved = __convolved
        return value

    def names(self, keep=False):
        names = Factor.names(self)

        _keep = []
        for i, name in enumerate(names):
            if name.find(downtime) < 0:
                _keep.append(i)

        if not keep:
            return [names[i] for i in _keep]
        else:
            return [names[i] for i in _keep], _keep

    def fromiterator(self, iterator, delimiter=','):
        """
        Determine an ExperimentalFactor from an iterator
        """

        if type(iterator) is types.StringType:
            iterator = csv.reader(file(iterator), delimiter=delimiter)
        elif type(iterator) is types.FileType:
            iterator = csv.reader(iterator, delimiter=delimiter)

        self.events = {}
        for row in iterator:
            if not self.delta:
                eventtype, start, end = row
                if not self.events.has_key(eventtype):
                    self.events[eventtype] = Events(name=eventtype)
                self.events[eventtype].append(float(start), float(end)-float(start), height=1.0)
            else:
                eventtype, start = row
                if not self.events.has_key(eventtype):
                    self.events[eventtype] = Events(name=eventtype)
                self.events[eventtype].append(float(start), self.dt, height=1.0/self.dt)


class ExperimentalFormula(Formula):

    """
    A formula with no intercept.
    """


    def __mul__(self, other, nested=False):
        return ExperimentalFormula(Formula.__mul__(self, other))

    def __add__(self, other):
        return ExperimentalFormula(Formula.__add__(self, other))

    def __call__(self, time, namespace=namespace, **keywords):

        allvals = []

        for term in self.terms:
            if not hasattr(term, 'IRF'):
                val = term(time=time, namespace=namespace, **keywords)
            else:
                val = term(time=time, namespace=namespace, **keywords)
                      
            if val.ndim == 1:
                val.shape = (1, val.shape[0])
            allvals.append(val)

        tmp = N.concatenate(allvals)
        names, keep = self.names(keep=True)

        return N.array([tmp[i] for i in keep])

    def names(self, keep=False):
        names = Formula.names(self)
        
        _keep = []
        for i, name in enumerate(names):
            name = names[i]
            if name.find(downtime) < 0:
                _keep.append(i)

        if not keep:
            return [names[i] for i in _keep]
        else:
            return [names[i] for i in _keep], _keep

class InterpolatedConfound(TimeFunction):

    times = traits.Any()
    values = traits.Any()

    def __init__(self, **keywords):
        TimeFunction.__init__(self, **keywords)
        if len(N.asarray(self.values).shape) == 1:
            self.f = interp1d(self.times, self.values, bounds_error=0)
            self.nout = 1
        else:
            self.f = []
            values = N.asarray(self.values)
            for i in range(values.shape[0]):
                f = interp1d(self.times, self.values[:,i], bounds_error=0)
                self.f.append(f)
            self.nout = values.shape[0]
            
    def __call__(self, time=None, **extra):
        columns = []

        if self.nout == 1:
            columns.append(self.f(time))
        else:
            if type(self.f) in [types.ListType, types.TupleType]:
                for f in self.f:
                    columns.append(f(time))
            else:
                columns = self.f(time)

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

class Stimulus(TimeFunction):

    times = traits.Any()
    values = traits.Any()

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

class DeltaFunction(TimeFunction):

    """
    A square wave approximate delta function returning
    1/dt in interval [start, start+dt).
    """

    start = traits.Trait(0.0, desc='Beginning of delta function approximation.')
    dt = traits.Float(0.02, desc='Width of delta function approximation.')

    def __call__(self, time=None, **extra):
        return N.greater_equal(time, self.start) * N.less(time, self.start + self.dt) / self.dt

class SplineConfound(FunctionConfound):

    """
    A natural spline confound with df degrees of freedom.
    """
    
    df = traits.Int(4)
    knots = traits.List()

    def __init__(self, **keywords):

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
                return N.power(time, 3.0) * N.greater(time, a) * N.less_equal(time, b)
            return _spline

        for i in range(len(self.knots) - 1):
            self.fn.append(_getspline(self.knots[i], self.knots[i+1]))

        self.nout = self.df
