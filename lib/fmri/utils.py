from numpy import *
import numpy.dft as FFT
import bisect
import scipy.interpolate

def bsearch(time, x, which=bisect.bisect_left, check_equality=False):
    """
    A binary search for time values, x can be a numpy.
    """

    _x = array(x)

    _time = list(time)
    if len(_x.shape) >= 1:
        _shape = _x.shape
        _x.shape = (product(_shape),)
        value = zeros(_x.shape)
        for i in range(_x.shape[0]):
            try:
                value[i] = _time.index(_x[i])
            except:
                value[i] = which(_time, _x[i])
            if value[i] < len(_time):
                if _x[i] < _time[value[i]]:
                    value[i] -= 1
        value.shape = _shape
    else:
        try:
            value = _time.index(_x)
        except:
            value = which(_time, _x)
            if value < len(_time):
                if _x < _time[value]:
                    value -= 1
    return array(value)

def fwhm2sigma(fwhm):
    return fwhm / sqrt(8 * log(2))

def sigma2fwhm(sigma):
    return sigma * sqrt(8 * log(2))

def ECDF(values):
    _values = list(values)
    _values.sort()
    n = 1. * len(_values)
    def fn(x):
        value = bsearch(_values, x, which=bisect.bisect_right, check_equality=True) / n
        return value
    return fn

def monotone_fn_inverter(fn, x, vectorized=True, **keywords):
    if vectorized:
        y = fn(x, **keywords)
    else:
        y = []
        for _x in x:
            y.append(fn(_x, **keywords))
        y = array(y)
    return LinearInterpolant(y, x, sorted=False)


def norm(X, p=2, axis=0):
    return pow(add.reduce(X**p,axis=axis),1.0/p)

def inv2(X):
    _X = abs(X)
    return greater(_X, 0) / (X + equal(abs(_X), 0))

def gradient(X, axis=0):
    ndim = len(X.shape)
    if ndim == 1:
        f = gradient1d
    elif ndim ==  2:
        f = gradient2d
    elif ndim == 3:
        f = gradient3d
    else:
        raise NotImplementedError, 'only upto 3d arrays are implemented'
    return f(X, axis=axis)

def gradient3d(X, axis=0):
    value = zeros(X.shape, Float)

    if axis == 0:
        value[1:-1,:,:] = (X[2:,:,:] - X[0:-2,:,:]) / 2.
        value[0,:,:] = X[1,:,:] - X[0,:,:]
        value[-1,:,:] = X[-1,:,:] - X[-2,:,:]
    elif axis == 1:
        value[:,1:-1,:] = (X[:,2:,:] - X[:,0:-2,:]) / 2.
        value[:,0,:] = X[:,1,:] - X[:,0,:]
        value[:,-1,:] = X[:,-1,:] - X[:,-2,:]
    elif axis == 2:
        value[:,:,1:-1] = (X[:,:,2:] - X[:,:,0:-2]) / 2.
        value[:,:,0] = X[:,:,1] - X[:,:,0]
        value[:,:,-1] = X[:,:,-1] - X[:,:,-2]

    return value

def gradient1d(X, axis=0):
    value = zeros(X.shape, Float)

    value[1:-1] = (X[2,:] - X[0:-2,:]) / 2.
    value[0] = X[1] - X[0]
    value[1] = X[-1] - X[-2]
    return value

def gradient2d(X, axis=0):
    value = zeros(X.shape, Float)

    if axis == 0:
        value[1:-1,:] = (X[2:,:] - X[0:-2,:]) / 2.
        value[0,:] = X[1,:] - X[0,:]
        value[-1,:] = X[-1,:] - X[-2,:]
    elif axis == 1:
        value[:,1:-1] = (X[:,2] - X[:,0:-2]) / 2.
        value[:,0] = X[:,1] - X[:,0]
        value[:,-1] = X[:,-1] - X[:,-2]

    return value

def inv(X):
    return greater(X, 0) / (X + less_equal(X, 0))

class StepFunction:
    '''A basic step function: values at the ends are handled in the simplest way possible: everything to the left of x[0] is set to y[0]; everything to the right of x[-1] is set to y[-1].
    
    >>>
    >>> from neuromaging.fmri.utils import StepFunction as step
    >>> from numpy import *
    >>>
    >>> x = arange(20)
    >>> y = arange(20)
    >>>
    >>> f = step(x, y)
    >>>
    >>> print f(3.2)
    3
    >>> print f([[3.2,4.5],[24,-3.1]])
    [[ 3  4]
     [19  0]]
    >>>

    '''

    def __init__(self, x, y, sorted=False):
            
        x = array(x)
        y = array(y)
        if x.shape != y.shape:
            raise ValueError, 'in StepFunction: x and y do not have the same shape!'
        if len(x.shape) != 1:
            raise ValueError, 'in StepFunction: x must be 1-dimensional'
        if len(y.shape) != 1:
            raise ValueError, 'in StepFunction: y must be 1-dimensional'
        self.n = x.shape[0]

        if not sorted:
            indices = argsort(array(x))
            self.x = take(x, indices)
            self.y = take(y, indices)
        else:
            self.x = x
            self.y = y

        self.mintime = x[0]
        self.maxtime = x[-1]

    def __call__(self, time):

        index_t = clip(bsearch(self.x, time, check_equality=True), 0, self.x.shape[0] - 1)
        if index_t.shape is ():
            return self.y[index_t]
        else:
            tmp = take(self.y, index_t)
            return tmp
    
        # Note that the first time is kind of useless here, as the call
        # assumes that the function takes on y[0] for t<self.x[1].
        # Also, the function is y[-1] for t >= self.x[-1].


def LinearInterpolant(x, y):
    return scipy.interpolate.InterpolatedUnivariateSpline(x,y,k=1)

class WaveFunction:
    def __init__(self, start, duration, height):
        self.start = start
        self.duration = duration
        self.height = height

    def __call__(self, time):
        return greater_equal(time, self.start) * less(time, self.start + self.duration) * self.height

# return the convolution (as a StepFunction) of two functions over interval,
# with grid size dt

def ConvolveFunctions(fn1, fn2, interval, dt, padding_f=0.1, offset1=0, offset2=0, normalize=[0,1]):
    """
    Convolve fn1 with fn2 -- where fn1 may return a multidimensional output.
    """
    
    ltime = max(interval) - min(interval)
    time = arange(min(interval), max(interval) + padding_f * ltime, dt)

    _fn1 = array(fn1(time  + offset1))
    _fn2 = array(fn2(time + offset2))

    if normalize[0]:
        _fn1 = _fn1 / sqrt(add.reduce(_fn1**2))
    _fft1 = FFT.real_fft(_fn1)

    if normalize[1]:
        _fn2 = _fn2 / sqrt(add.reduce(_fn2**2))
    _fft2 = FFT.real_fft(_fn2)
    value = FFT.inverse_real_fft(_fft1 * _fft2)
    _minshape = time.shape[0]
    time = time[0:_minshape]
    value = value[0:_minshape]
    
    if len(value.shape) == 2:
        fns = []
        for i in range(value.shape[0]):
            fns.append(LinearInterpolant(time, value[i]))
        return fns
    else:
        return LinearInterpolant(time, value)

class CutPoly:
    def __init__(self, trange=None, power=1, range_type='lower'):
        self.power = power
        if trange is not None:
            self.trange = trange
        self.range_type = range_type

    def __call__(self, time):
        test = None
        if hasattr(self, 'trange'):
            if self.range_type == 'lower':
                test = greater_equal(time, self.trange)
            elif self.range_type == 'upper':
                test = less(time, self.trange)
            else:
                test = greater_equal(time, self.trange[0]) * less_equal(time, self.trange[1])
        if test is None:
            return pow(time, self.power)
        else:
            return pow(time, self.power) * test
        
def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
