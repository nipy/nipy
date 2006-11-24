import csv, types, copy
import numpy as N

from neuroimaging.modalities.fmri.functions import TimeFunction, Events
from scipy.sandbox.models.formula import factor, quantitative, formula, term
#from enthought.traits.api import false, HasTraits, Str
namespace = {}
downtime = 'None/downtime'

class ExperimentalRegressor(object):
    

    def __init__(self, convolved=False, namespace=namespace, termname='term'):
	namespace[termname] = self
        self.__c = convolved
	self.termname = termname

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
	    self._funcunconv = self.func
	    
        if self.convolved:
            self.name = self._convolved.name
	    self.func = self._convolved
        else:
            self.name = self._nameunconv
	    self.func = self._funcunconv

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
                return term.names(self)

    def _convolved__call__(self, time, **keywords):
	"""
	Call convolved function.
	"""
	v = []
	for _func in self._convolved_func:
	    try:
		v.append(_func(time, **keywords))
	    except:
		for __func in _func:
		    v.append(__func(time, **keywords))
	return N.array(v)

    def convolve(self, IRF):

        self.convolved = False
        self._convolved_func = IRF.convolve(self)
       
        name = []
        for hrfname in IRF.names:
            for termname in self.names():
                name.append('(%s%%%s)' % (hrfname, termname))
 
        self._convolved = ExperimentalQuantitative(name, self._convolved__call__, termname='(%s%%%s)' % (hrfname, self.termname))
        self.convolved = True

        return self

    def astimefn(self, slice=None):
        """
        Return a TimeFunction object that can be added, subtracted, etc.
        """
	#FIXME: should be a better way to determine this
	testdata = self(N.arange(10))
	if len(testdata.shape) == 2:
	    nout = testdata.shape[0]
	else: nout = 1
        return TimeFunction(fn=self, slice=slice, nout=nout)


class ExperimentalQuantitative(ExperimentalRegressor, quantitative):
    """
    Generate a regressor that is a function of time
    based on a function fn.
    """

    def __init__(self, name, fn, termname=None, **keywords):

        self.func = fn
        self.name = name
        if termname is None:
            termname = name
     
        ExperimentalRegressor.__init__(self, termname=termname, **keywords)

        test = N.array(self.func(time=N.array([4.0,5.0,6])))
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

        quantitative.__init__(self, names, func=fn, termname=termname, **keywords)
            
class Time(ExperimentalQuantitative):
    
    def __call__(self, time, **ignored):
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

class ExperimentalFactor(ExperimentalRegressor, factor):
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
        factor.__init__(self, name, keys)
        self._event_keys = self.events.keys()
        self._event_keys.sort()
        namespace[name] = self

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

        def factor_func(time, namespace=namespace, j=j,
                obj=self, **ignored):
            _c = obj.convolved
            obj.convolved = False
            v = obj(time, namespace=namespace)[j]
            obj.convolved = _c
            return [N.squeeze(v) * 1.]

        name = '%s[%s]' % (self.termname, `key`)
        return ExperimentalQuantitative(name, factor_func)

    def __call__(self, time, includedown=False, convolved=None, **kw):
        if convolved is not None:
            __convolved, self.convolved = self.convolved, convolved
        else:
            __convolved = self.convolved

        if not self.convolved:
            value = []
            keys = self._event_keys
            for level in keys:
                value.append(N.squeeze(self.events[level](time, **kw)))
            if includedown:
                s = N.add.reduce(value)

                keys = keys + [downtime]
                which = N.argmax(value, axis=0)
                which = N.where(s, which, keys.index(downtime))
		tmp, self.namespace = self.namespace, {self.termname:[keys[w] for w in which]}
                value = factor.__call__(self)
		self.namespace = tmp
            else:
                value = N.asarray(value)
        else:
            if hasattr(self, '_convolved'):
                value = self._convolved(time, **kw)
            else:
                raise ValueError, 'no IRF defined for factor %s' % self.name
        self.convolved = __convolved
        return value

    def names(self, keep=False):
        names = factor.names(self)

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


class ExperimentalFormula(formula):
    """
    A formula with no intercept.
    """


    def __mul__(self, other, nested=False):
        return ExperimentalFormula(formula.__mul__(self, other))

    def __add__(self, other):
        return ExperimentalFormula(formula.__add__(self, other))

    def __call__(self, time, **keywords):

        allvals = []

        for t in self.terms:
	    t.namespace = self.namespace
	    val = t(time, **keywords)

            if val.ndim == 1:
                val.shape = (1, val.shape[0])
            allvals.append(val)

        tmp = N.concatenate(allvals)
        names, keep = self.names(keep=True)

        return N.array([tmp[i] for i in keep])

    def names(self, keep=False):
        names = formula.names(self)

        _keep = []
        for i, name in enumerate(names):
            name = names[i]
            if name.find(downtime) < 0:
                _keep.append(i)

        if not keep:
            return [names[i] for i in _keep]
        else:
            return [names[i] for i in _keep], _keep

