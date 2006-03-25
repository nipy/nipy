import numpy as N
import numpy.dft as FFT
import bisect
import scipy.interpolate
from neuroimaging.statistics.utils import StepFunction

def fwhm2sigma(fwhm):
    """
    Convert a FWHM value to sigma in a Gaussian kernel.
    """
    return fwhm / N.sqrt(8 * log(2))

def sigma2fwhm(sigma):
    """
    Convert a sigma in a Gaussian kernel to a FWHM value.
    """
    return sigma * N.sqrt(8 * log(2))

def monotone_fn_inverter(fn, x, vectorized=True, **keywords):
    """
    Given a monotone function x (no checking is done to verify montonocity)
    and a set of x values, return an linearly
    interpolated approximation to its inverse from its values on x.
    """

    if vectorized:
        y = fn(x, **keywords)
    else:
        y = []
        for _x in x:
            y.append(fn(_x, **keywords))
        y = N.array(y)
    return LinearInterpolant(y, x, sorted=False)

def norm(X, p=2, axis=0):
    return N.pow(N.add.reduce(X**p,axis=axis),1.0/p)

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
    value = N.zeros(X.shape, N.Float)

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
    value = N.zeros(X.shape, N.Float)

    value[1:-1] = (X[2,:] - X[0:-2,:]) / 2.
    value[0] = X[1] - X[0]
    value[1] = X[-1] - X[-2]
    return value

def gradient2d(X, axis=0):
    value = N.zeros(X.shape, N.Float)

    if axis == 0:
        value[1:-1,:] = (X[2:,:] - X[0:-2,:]) / 2.
        value[0,:] = X[1,:] - X[0,:]
        value[-1,:] = X[-1,:] - X[-2,:]
    elif axis == 1:
        value[:,1:-1] = (X[:,2] - X[:,0:-2]) / 2.
        value[:,0] = X[:,1] - X[:,0]
        value[:,-1] = X[:,-1] - X[:,-2]

    return value

class LinearInterpolant:
    """
    A little wrapper around scipy.interpolate call to force
    the interpolant to take a keywords argument \'time=\'.
    """

    def __init__(self, x, y, k=1):
        self.f = scipy.interpolate.interp1d(x, y)

    def __call__(self, time=None, **keywords):
        return self.f(time)

class WaveFunction:
    def __init__(self, start, duration, height):
        self.start = start
        self.duration = duration
        self.height = height

    def __call__(self, time):
        return N.greater_equal(time, self.start) * N.less(time, self.start + self.duration) * self.height

# return the convolution (as a StepFunction) of two functions over interval,
# with grid size dt

def ConvolveFunctions(fn1, fn2, interval, dt, padding_f=0.1, offset1=0, offset2=0, normalize=[0,1]):
    """
    Convolve fn1 with fn2 -- where fn1 may return a multidimensional output.
    """
    
    ltime = max(interval) - min(interval)
    time = N.arange(min(interval), max(interval) + padding_f * ltime, dt)

    _fn1 = N.array(fn1(time  + offset1))
    _fn2 = N.array(fn2(time + offset2))

    if normalize[0]:
        _fn1 = _fn1 / N.sqrt(N.add.reduce(_fn1**2))
    _fft1 = FFT.real_fft(_fn1)

    if normalize[1]:
        _fn2 = _fn2 / N.sqrt(N.add.reduce(_fn2**2))
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
                test = N.less(time, self.trange)
            else:
                test = N.greater_equal(time, self.trange[0]) * N.less_equal(time, self.trange[1])
        if test is None:
            return N.pow(time, self.power)
        else:
            return N.pow(time, self.power) * test
        
def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
