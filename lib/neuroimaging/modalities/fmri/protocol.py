import csv, types, copy
import numpy as N

from neuroimaging.modalities.fmri.functions import TimeFunction, Events
from scipy.sandbox.models.formula import Factor, Quantitative, Formula, Term

namespace = {}
downtime = 'None/downtime'

class ExperimentalRegressor(object):

    
    def __init__(self, convolved=False):
        self.__c = convolved

    # a toggle to determine whether we
    # want to think of the factor as convolved or not
    # i.e. for plotting
    # fixme: this is ugly, but was confusing with traits
    # we can put it back to being a trait when my brain
    # sorts itself out -- Timl
    def _get_c(self):  return self.__c
    def _set_c(self, value):  self.__c = value; self._convolved_changed()
    def _del_c(self): del self.__c
    convolved = property(_get_c, _set_c, _del_c)

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
            print self.convolved
            return self._convolved.names()
        else:
            if hasattr(self, '_nameunconv'):
                return self._nameunconv
            else:
                return Term.names(self)

    def convolve(self, IRF):

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
        fn = self._fromiterator(iterator)
        ExperimentalQuantitative.__init__(self, name, fn, **keywords)
  
    def _fromiterator(self, iterator, delimiter=','):
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

        # self.name doesn't exist
        self.events = Events(name=self.name)

        for row in iterator:
            try:
                start, end, height = row
            except:
                start, end = row
                height = 1.0
                pass

            #event type doesn't exist
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
    
    def __init__(self, name, iterator, convolved=False, delta=True, dt=0.02):
        ExperimentalRegressor.__init__(self, convolved)
        self.delta = delta
        self.dt = dt
        
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
            raise KeyError, 'key not found'            

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
                dt = float(end) - float(start)
                height = 1.0
            else:
                eventtype, start = row
                dt = self.dt
                height = 1.0/self.dt
            if not self.events.has_key(eventtype):
                self.events[eventtype] = Events(name=eventtype)
            self.events[eventtype].append(float(start), dt, height=height)


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

