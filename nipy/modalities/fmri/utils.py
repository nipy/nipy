__docformat__ = 'restructuredtext'

import numpy as np
import numpy.fft as FFT
import scipy.interpolate

class LinearInterpolant(object):
    """
    A little wrapper around scipy.interpolate call to force
    the interpolant to take a keywords argument \'time=\'.
    """

    def __init__(self, x, y, fill_value=0.):
        """
        :Parameters:
            `x` : numpy.ndarray
                A 1D array of monotonically increasing real values. x cannot 
                include duplicate values (otherwise f is overspecified)
            `y` : numpy.ndarray
                An N-D array of real values. y's length along the interpolation
                axis must be equal to the length of x.
            `fill_value` : float
                If provided, then this value will be used to fill in for requested
                points outside of the data range.
                
        :Prerequisite: len(x) == len(y)
        """
        self.f = scipy.interpolate.interp1d(x, y, bounds_error=0, fill_value=fill_value)

    def __call__(self, time=None, **keywords):
        """
        :Parameters:
            `time` : TODO
                TODO
            `keywords` : dict
                Keyword arguments are discarded.
                
        :Returns: TODO
        """
        return self.f(time)

class WaveFunction(object):
    """
    A square wave function of a specified start, duration and height.
    f(t) = height if (start <= t < start + duration), 0 otherwise
    """

    def __init__(self, start, duration, height):
        """
        :Parameters:
            `start` : float
                The time of the rising edge of the square wave.
            `duration` : float
                The width of the square wave
            `height` : float
                The height of the square wave
        """
        self.start = start
        self.duration = duration
        self.height = float(height)

    def __call__(self, time):
        """
        :Parameters:
            `time` : float or numpy.ndarray
                A time value or values for the function to be evaluated at.
        
        :Returns: ``float`` or ``numpy.ndarray``
        """
        return np.greater_equal(time, self.start) * np.less(time, self.start + self.duration) * self.height

# return the convolution (as a StepFunction) of two functions over interval,
# with grid size dt

def ConvolveFunctions(fn1, fn2, interval, dt, padding_f=0.1, normalize=(0, 0)):
    """
    Convolve fn1 with fn2 -- where fn1 may return a multidimensional output.
    
    :Parameters:
        `fn1` : TODO
            TODO
        `fn2` : TODO
            TODO
        `interval` : TODO
            TODO
        `dt` : TODO
            TODO
        `padding_f` : float
            TODO
        `normalize` : TODO
            TODO
            
    :Returns: TODO
    """

    max_interval, min_interval = max(interval), min(interval)
    ltime = max_interval - min_interval
    time = np.arange(min_interval, max_interval + padding_f * ltime, dt)

    _fn1 = np.array(fn1(time))
    _fn2 = np.array(fn2(time))

    if normalize[0]:
        _fn1 /= np.sqrt(np.add.reduce(_fn1**2))
    _fft1 = FFT.rfft(_fn1)

    if normalize[1]:
        _fn2 /= np.sqrt(np.add.reduce(_fn2**2))
    _fft2 = FFT.rfft(_fn2)

    value = FFT.irfft(_fft1 * _fft2)
    _minshape = min(time.shape[0], value.shape[-1])
    time = time[0:_minshape]
    value = value[0:_minshape]
    
    if len(value.shape) == 2:
        fns = []
        for i in range(value.shape[0]):
            fns.append(LinearInterpolant(time + min_interval, value[i]))

        return fns
    else:
        newf = LinearInterpolant(time + min_interval, value)
        return newf

class CutPoly(object):
    """
    A polynomial function of the form f(t) = t^n with an optionally fixed
    time range on which the function exists.
    """

    def __init__(self, power, trange=(None, None)):
        """
        :Paramters:
            `power` : float
                f(t) = t^power
            `trange` : (float or None, float or None)
                A tuple with the upper and lower bound of the function.
                None signifies no boundary. Default = (None, None)
        """
        self.power = power
        self.trange = trange

    def __call__(self, time):
        """
        :Parameters:
            `time` : float or numpy.ndarray
                A time value or values for the function to be evaluated at.            
        
        :Returns: ``float`` or ``numpy.ndarray``
        """
        test = np.ones(np.asarray(time).shape)
        lower, upper = self.trange
        
        if lower is not None:
            test *= np.greater_equal(time, lower)
        if upper is not None:
            test *= np.less(time, upper)
        return np.power(time, self.power) * test
        
        
def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
