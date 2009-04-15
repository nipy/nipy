"""
TODO
"""

__docformat__ = 'restructuredtext'

import numpy as np

from nipy.modalities.fmri.utils import ConvolveFunctions, WaveFunction

class Filter(object):
    '''
    Takes a list of impulse response functions (IRFs): main purpose is to
    convolve a functions with each IRF for Design. The class assumes the range
    of the filter is effectively 50 seconds, can be changed by setting tmax --
    this is just for the __mul__ method for convolution.
    '''


    def __init__(self, IRF, names, dt=0.02, tmin=-10., tmax=500.):
        """
        :Parameters:
            `IRF` : TODO
                TODO
            `names` : TODO
                TODO
            `dt` : float
                TODO
            `tmin` : float
                TODO
            `tmax` : float
                TODO
        """
        self.IRF = IRF
        self.names = names
        self.dt = dt
        self.tmin = tmin
        self.tmax = tmax
        try:
            self.n = len(self.IRF)
        except:
            self.n = 1

    def __getitem__(self, i):
        """
        :Parameters:
            `i` : int
                TODO
        
        :Returns: `Filter`
        
        :Raises ValueError: if ``i`` is not an int
        :Raises IndexError: if ``i`` is not a valid index
        """
        if not isinstance(i, int):
            raise ValueError, 'integer needed'
        if self.n == 1:
            if i != 0:
                raise IndexError, 'invalid index'
            try:
                IRF = self.IRF[0]
            except:
                IRF = self.IRF
            return Filter(IRF, names=[self.names[0]])
        else:
            return Filter(self.IRF[i], names=[self.names[i]])


    def __add__(self, other):
        """
        Take two Filters with the same number of outputs and create a new one
        whose IRFs are the sum of the two.


        :Returns: `Filter`
        """
        if self.n != other.n:
            raise ValueError, 'number of dimensions in Filters must agree'
        newIRF = []
        for i in range(self.n):
            def curfn(time):
                return self.IRF[i](time) + other.IRF[i](time)
            newIRF.append(curfn)
        return Filter(newIRF)

    def __mul__(self, other):
        """
        Take two Filters with the same number of outputs and create a new one
        whose IRFs are the convolution of the two.
        """
        if self.n != other.n:
            raise ValueError, 'number of dimensions in Filters must agree'
        newIRF = []
        interval = (self.tmin, self.tmax + other.tmax)
        for i in range(self.n):
            curfn = ConvolveFunctions(self.IRF[i], other.IRF[i], interval,
                                      self.dt)
            newIRF.append(curfn)
        return Filter(newIRF)

    def convolve(self, fn, interval=None, dt=None):
        """
        Take a (possibly vector-valued) function fn of time and return
        a linearly interpolated function after convolving with the filter.
        """
        if dt is None:
            dt = self.dt
        if interval is None:
            interval = [self.tmin, self.tmax]
        if self.n > 1:
            value = []
            for _IRF in self.IRF:
                value.append(ConvolveFunctions(fn, _IRF, interval, dt))
            return value
        else:
            return ConvolveFunctions(fn, self.IRF, interval, dt)

    def __call__(self, time):
        """
        Return the values of the IRFs of the filter.
        """
        if self.n > 1:
            value = np.zeros((self.n,) + time.shape)
            for i in range(self.n):
                value[i] = self.IRF[i](time)
        else:
            value = self.IRF(time)
        return value

class GammaDENS:
    """
    A class for a Gamma density which knows how to differentiate itself.

    By default, normalized to integrate to 1.
    """
    def __init__(self, alpha, nu, coef=1.0):
        """
        :Parameters:
            `alpha` : TODO
                TODO
            `nu` : TODO
                TODO
            `coef` : float
                TODO
        """
        self.alpha = alpha
        self.nu = nu
##        self.coef = nu**alpha / scipy.special.gamma(alpha)
        self.coef = coef

    def __str__(self):
        """
        :Returns: ``string``
        """
        return '<GammaDENS:alpha:%03f, nu:%03f, coef:%03f>' % (self.alpha,
                                                               self.nu,
                                                               self.coef)

    def __repr__(self):
        """
        :Returns: ``string``
        """
        return self.__str__()

##     def __mul__(self, const):
##         self.coef = self.coef * const
##         return self
    
    def __call__(self, x):
        '''Evaluate the Gamma density.'''
        _x = x * np.greater_equal(x, 0)
        return self.coef * np.power(_x, self.alpha-1.) * np.exp(-self.nu*_x)

    def deriv(self, const=1.):
        '''
        Differentiate a Gamma density. Returns a GammaCOMB that can evaluate
        the derivative.
        '''
        return GammaCOMB([[const*self.coef*(self.alpha-1),
                           GammaDENS(self.alpha-1., self.nu)],
                          [-const*self.coef*self.nu,
                           GammaDENS(self.alpha, self.nu)]])

class GammaCOMB:
    """
    TODO
    """
    
    def __init__(self, fns):
        """
        :Parameters:
            `fns` : TODO
                TODO
        """
        self.fns = fns

    def __mul__(self, const):
        """
        :Parameters:
            `const` : TODO
                TODO
                
        :Returns: `GammaCOMB`
        """
        fns = []
        for fn in self.fns:
            fns.append([fn[0] * const, fn[1]])
        return GammaCOMB(fns)

    def __add__(self, other):
        """
        :Parameters:
            `other` : TODO
                TODO

        :Returns: `GammaCOMB`
        """
        fns = self.fns + other.fns
        return GammaCOMB(fns)

    def __call__(self, x):
        """
        :Parameters:
            `x` : TODO
                TODO
        
        :Returns: TODO
        """
        value = 0
        for coef, fn in self.fns:
            value = value + coef * fn(x)
        return value

    def deriv(self, const=1.):
        """
        :Parameters:
            `const` : float
                TODO
                
        :Returns: `GammaCOMB`
        """
        fns = []
        for coef, fn in self.fns:
            comb = fn.deriv(const=const)
            comb.fns[0][0] = comb.fns[0][0] * coef
            comb.fns[1][0] = comb.fns[1][0] * coef
            fns = fns + comb.fns
        return GammaCOMB(fns)
    
class GammaHRF(Filter):
    """
    A class that represents the Gamma basis in SPM: i.e. the filter is a
    collection of a certain number of Gamma densities. Parameters are
    specified as a kx2 matrix for k Gamma functions.
    """

    def __init__(self, parameters):
        """
        :Parameters:
            `parameters` : TODO
                TODO
        """
        fns = [GammaDENS(alpha, nu) for alpha, nu in parameters]
        Filter.__init__(self, fns)

    def deriv(self, const=1.):
        """
        :Parameters:
            `const` : float
                TODO
        
        :Returns: TODO
        """
        return [fn.deriv(const=const) for fn in self.IRF]
    
class FIR(Filter):
    """
    A class for FIR filters: i.e. the filter is a collection of square waves.
    Parameters (start and duration) are specified as a kx2 matrix for k square
    waves.

    >>> GUI = True
    >>> from nipy.modalities.fmri import filters
    >>> from pylab import *
    >>> from numpy import *
    >>> parameters = array([[1., 2.], [2., 5.], [4., 8.]])
    >>> IRF = filters.FIR(parameters)
    >>> _ = plot(arange(0, 15, 0.1), sum(IRF(arange(0, 15, 0.1)), axis=0))
    >>> ylab = ylabel('Filters')
    >>> xlab = xlabel('Time (s)')
    >>> show()

    """

    def __init__(self, parameters):
        """
        :Parameters:
            `parameters` : TODO
                TODO
        """
        fns = [WaveFunction(start, duration, 1.0) for
               (start, duration) in parameters]
        Filter.__init__(self, fns, names="FIR")

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
