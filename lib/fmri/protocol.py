import csv, string, types
import enthought.traits as traits
import numpy as N

from regressors import Events, SplineConfound
from formula import Factor, Quantitative, Formula

namespace = {}
downtime = 'None/downtime'

class ExperimentalRegressor(traits.HasTraits):

    def __add__(self, other):
        other = ExperimentalFormula(other)
        return other + self

    def __mul__(self, other):
        other = ExperimentalFormula(other)
        return other * self

    def convolve(self, IRF):
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
            for varname in self.names():
                name.append('(%s**%s)' % (hrfname, varname))
        return ExperimentalQuantitative(name, _f, varname='(%s**%s)' % (hrfname, self.varname))

class ExperimentalQuantitative(ExperimentalRegressor, Quantitative):
    """
    Generator a regressor that is a function of time
    based on a function fn.
    """

    def __init__(self, name, fn, varname=None, **keywords):

        self.fn = fn
        self.name = name
        if varname is None:
            varname = name
        namespace[varname] = self
            
        test = self.fn(N.array([4.0]))
        n = len(test)

        if n > 1:
            if type(name) in [type([]), type(())]:
                names = name
            else:
                names = ['(%s:%d)' % (name, i) for i in range(n)]
        else:
            names = name
        Quantitative.__init__(self, names, _fn=fn, varname=varname, **keywords)
        
    def __call__(self, time=None, namespace=namespace, **keywords):
        return self.fn(time, **keywords)

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
        Determine an ExperimentalFactor which returns rows of the form:

        (start, stop, height)

        Here, Type is a hashable object and Start and Stop are floats.
        The fourth being an optional
        float for the height during the interval [Start,Stop] which defaults to 1.
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
        traits.HasTraits.__init__(self, **keywords)
        self.fromiterator(iterator)
        keys = self.events.keys() + [downtime]
        Factor.__init__(self, name, keys)
        
    def __call__(self, time=None, namespace=None, **keywords):
        value = []
        keys = self.events.keys()
        for level in keys:
            value.append(N.squeeze(self.events[level](time,
                                                      namespace=namespace,
                                                      **keywords)))
        s = N.add.reduce(value)

        keys = keys + [None]
        which = N.argmax(value, axis=0)
        which = N.where(s, which, keys.index(None))
        return Factor.__call__(self, namespace={self.varname:[keys[w] for w in which]})

    def fromiterator(self, iterator, delimiter=','):
        """
        Determine an ExperimentalFactor which returns rows of the form:

        Type,Start,Stop

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

    def __call__(self, time=None, namespace=namespace, **keywords):

        allvals = []

        for var in self.variables:
            if not hasattr(var, 'IRF'):
                val = var(time=time, namespace=namespace, **keywords)
            else:
                val = var(time=time, namespace=namespace, convolved=True, **keywords)
                      
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


