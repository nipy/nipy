import csv, string, types
import enthought.traits as traits
import numpy as N

from regressors import Events, SplineConfound
from neuroimaging.statistics.formula import Factor, Quantitative, Formula

namespace = {}
downtime = 'None/downtime'

class ExperimentalRegressor(traits.HasTraits):

    convolved = traits.false # a toggle to determine whether we
                             # want to think of the factor as convolved or not
                             # i.e. for plotting

    def __add__(self, other):
        other = ExperimentalFormula(other)
        return other + self

    def __mul__(self, other):
        other = ExperimentalFormula(other)
        return other * self

    def convolve(self, IRF):
        self.IRF = IRF
        _fn = IRF.convolve(self)
        def _f(time=None, _fn=tuple(_fn), **keywords):
            v = []
            for __fn in _fn:
                try:
                    v.append(__fn(time, **keywords))
                except:
                    for ___fn in __fn:
                        v.append(___fn(time, **keywords))
            return N.array(v)
        
        name = []
        for hrfname in IRF.names:
            for termname in self.names():
                name.append('(%s**%s)' % (hrfname, termname))
        self._convolved = ExperimentalQuantitative(name, _f, termname='(%s**%s)' % (hrfname, self.termname))

class ExperimentalQuantitative(ExperimentalRegressor, Quantitative):
    """
    Generator a regressor that is a function of time
    based on a function fn.
    """

    def __init__(self, name, fn, termname=None, **keywords):

        ExperimentalRegressor.__init__(self, **keywords)
        self.fn = fn
        self.name = name
        if termname is None:
            termname = name
        namespace[termname] = self
            
        test = self.fn(N.array([4.0]))
        n = len(test)

        if n > 1:
            if type(name) in [type([]), type(())]:
                names = name
            else:
                names = ['(%s:%d)' % (name, i) for i in range(n)]
        else:
            names = name
        Quantitative.__init__(self, names, _fn=fn, termname=termname, **keywords)
        
    def __call__(self, time=None, namespace=namespace, **keywords):
        if not self.convolved:
            v = self.fn(time=N.arange(20), **keywords)
            return self.fn(time=time, **keywords)
        else:
            if hasattr(self, '_convolved'):
                return self._convolved(time=time, namespace=namespace, **keywords)
            else:
                raise ValueError, 'term %s has not been convolved with an HRF' % self.name
            
class ExperimentalStepFunction(ExperimentalQuantitative):
    """
    Return a step function from an iterator returing tuples

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
            iterator = csv.reader(file(iterator))
        elif type(iterator) is types.FileType:
            iterator = csv.reader(iterator)

        self.event = Events(name=self.name)

        for row in iterator:
            try:
                start, end, height = row
            except:
                start, end = row
                height = 1.0
                pass
            
            if type(end) is type('0.0'):
                end = string.atof(end)
            if type(start) is type('0.0'):
                start = string.atof(start)
            if type(height) is type('0.0'):
                height = string.atof(height)
            self.event[eventtype].append(start, end-start, height=height)
        return self.event

class ExperimentalFactor(ExperimentalRegressor, Factor):
    """
    Return a factor that is a function of experimental time.
    """
    
    def __init__(self, name, iterator, **keywords):
        ExperimentalRegressor.__init__(self, **keywords)
        self.fromiterator(iterator)
        keys = self.events.keys() + [downtime]
        Factor.__init__(self, name, keys)
        
    def __call__(self, time=None, namespace=None, includedown=False, **keywords):
        if not self.convolved:
            value = []
            keys = self.events.keys()
            for level in keys:
                value.append(N.squeeze(self.events[level](time,
                                                          namespace=namespace,
                                                          **keywords)))
            if includedown:
                s = N.add.reduce(value)

                keys = keys + [downtime]
                which = N.argmax(value, axis=0)
                which = N.where(s, which, keys.index(downtime))
                return Factor.__call__(self, namespace={self.termname:[keys[w] for w in which]})
            else:
                return N.array(value)
        else:
            if hasattr(self, '_convolved'):
                return self._convolved(time=time, namespace=namespace, **keywords)
            else:
                raise ValueError, 'no IRF defined for factor %s' % self.name

    def names(self, keep=False):
        names = Factor.names(self)

        _keep = []
        for i in range(len(names)):
            name = names[i]
            if name.find(downtime) < 0:
                _keep.append(i)

        if not keep:
            return [names[i] for i in _keep]
        else:
            return [names[i] for i in _keep], _keep


    def fromiterator(self, iterator, delimiter=','):
        """
        Determine an ExperimentalFactor from an iterator
        which returns rows of the form:

        Type, Start, Stop

        Here, Type is a hashable object and Start and Stop are floats.
        The fourth being an optional
        float for the height during the interval [Start,Stop] which defaults to 1.
        """

        if type(iterator) is types.StringType:
            iterator = csv.reader(file(iterator))
        elif type(iterator) is types.FileType:
            iterator = csv.reader(iterator)

        self.events = {}
        for row in iterator:
            eventtype, start, end = row
            if not self.events.has_key(row[0]):
                self.events[eventtype] = Events(name=eventtype)

            if type(end) is type('0.0'):
                end = string.atof(end)
            if type(start) is type('0.0'):
                start = string.atof(start)
            self.events[eventtype].append(start, end-start, height=1.0)

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
                      
            allvals.append(val)

        tmp = N.concatenate(allvals)

        names, keep = self.names(keep=True)

        return N.array([tmp[i] for i in keep])

    def names(self, keep=False):
        names = Formula.names(self)
        
        _keep = []
        for i in range(len(names)):
            name = names[i]
            if name.find(downtime) < 0:
                _keep.append(i)

        if not keep:
            return [names[i] for i in _keep]
        else:
            return [names[i] for i in _keep], _keep



