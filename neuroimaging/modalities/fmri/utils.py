__docformat__ = 'restructuredtext'

import numpy as np
import numpy.fft as FFT
from scipy.interpolate import interp1d

# return the convolution linear interpolant of two functions over interval,
# with grid size dt

def convolve_functions(fn1, fn2, interval, dt, padding_f=0.1, normalize=(0, 0)):
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
            fns.append(interp1d(time + min_interval, value[i]))

        return fns
    else:
        newf = interp1d(time + min_interval, value)
        return newf

