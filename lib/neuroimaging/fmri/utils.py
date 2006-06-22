import numpy as N
import numpy.dft as FFT
import scipy.interpolate

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

class LinearInterpolant:
    """
    A little wrapper around scipy.interpolate call to force
    the interpolant to take a keywords argument \'time=\'.
    """

    def __init__(self, x, y, k=1, fill_value=0.):
        self.f = scipy.interpolate.interp1d(x, y, bounds_error=0, fill_value=0.)

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

def ConvolveFunctions(fn1, fn2, interval, dt, padding_f=0.1, normalize=[0,0]):
    """
    Convolve fn1 with fn2 -- where fn1 may return a multidimensional output.
    """
    
    ltime = max(interval) - min(interval)
    time = N.arange(min(interval), max(interval) + padding_f * ltime, dt)

    _fn1 = N.array(fn1(time))
    _fn2 = N.array(fn2(time))

    if normalize[0]:
        _fn1 = _fn1 / N.sqrt(N.add.reduce(_fn1**2))
    _fft1 = FFT.rfft(_fn1)

    if normalize[1]:
        _fn2 = _fn2 / N.sqrt(N.add.reduce(_fn2**2))

    _fft2 = FFT.rfft(_fn2)
    value = FFT.irfft(_fft1 * _fft2)
    _minshape = min(time.shape[0], value.shape[-1])
    time = time[0:_minshape]
    value = value[0:_minshape]
    
    if len(value.shape) == 2:
        fns = []
        for i in range(value.shape[0]):
            fns.append(LinearInterpolant(time + min(interval), value[i]))

        return fns
    else:
        newf = LinearInterpolant(time + min(interval), value)
        return newf

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
            return N.power(time, self.power)
        else:
            return N.power(time, self.power) * test
        
def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
