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

    def convolve(self):
        _fn = self.IRF.convolve(self)
        def _f(_fn=_fn, **keywords):
            
            return _fn(**keywords)
        name = []
        for hrfname in self.IRF.names:
            for varname in self.names():
                name.append('%s*%s' % (hrfname, varname))
        name = name
        return ExperimentalQuantitative(name, _f, varname='%s*%s' % (hrfname, self.varname))

class ExperimentalQuantitative(ExperimentalRegressor, Quantitative):
    """
    Generator a regressor that is a function of time
    based on a function fn.
    """

    def __init__(self, name, fn, **keywords):

        self.fn = fn
        self.name = name
        if not keywords.has_key('varname'):
            namespace[name] = self
        else:
            namespace[keywords['varname']] = self
            
        test = self.fn(N.array([4.0]))
        n = len(test)
        if n > 1:
            names = ['%s:%d' % (name, i) for i in range(n)]
        else:
            names = name
        Quantitative.__init__(self, names, fn, **keywords)
        
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

        names = self.names()
        keep = []
        for i in range(len(names)):
            name = names[i]
            print name
            if name.find(downtime) < 0:
                keep.append(i)

        return N.array([tmp[i] for i in keep])

if __name__ == '__main__':

    import hrf
     
    t = N.arange(0,300,1)

    p = ExperimentalFactor('pain', csv.reader(file('pain.csv')))
    print p.names()

    notconvolved = p(t)
    print notconvolved.shape

    p.IRF = hrf.HRF(deriv=True)
    p.convolve()

    pc = convolved = p(t)
    print convolved.shape

    drift = ExperimentalQuantitative('drift', SplineConfound(window=[0,300]))

    formula = p + drift * pc

    y = formula(t)
    import pylab
    for i in range(y.shape[0]):
        pylab.plot(t, y[i])
        pylab.figure()
#    pylab.show()

    d = formula.design(t)
    print d.shape
