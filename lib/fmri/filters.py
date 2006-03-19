from numpy import *
from utils import ConvolveFunctions, WaveFunction, StepFunction, LinearInterpolant
from numpy.linalg import generalized_inverse, singular_value_decomposition
import enthought.traits as traits

interpolant = LinearInterpolant

class Filter(traits.HasTraits):
    dt = traits.Float(0.2)
    tmax = traits.Float(500.0)
    delta = arange(-4.2,4.2,0.1)

    '''Takes a list of impulse response functions (IRFs): main purpose is to convolve a functions with each IRF for Design. The class assumes the range of the filter is effectively 50 seconds, can be changed by setting tmax -- this is just for the __mul__ method for convolution.'''

    def __init__(self, IRF, **keywords):
        traits.HasTraits.__init__(self, **keywords)
        self.IRF = IRF
        try:
            self.n = len(self.IRF)
        except:
            self.n = 1

    def __add__(self, other):
        """
        Take two Filters with the same number of outputs and create a new one
        whose IRFs are the sum of the two.
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
        interval = (0, self.tmax + other.tmax)
        for i in range(self.n):
            curfn = ConvolveFunctions(self.IRF[i], other.IRF[i], interval, self.dt)
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
            interval = [0, self.tmax]
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
            value = zeros((self.n,) + time.shape, Float)
            for i in range(self.n):
                value[i] = self.IRF[i](time)
        else:
            value = self.IRF(time)
        return value

    def deltaPCA(self, delta, fn=None, dt=None, tmax=50., lower=-15.0, spectral=False):
        '''
        Perform an expansion of fn, shifted over the values in delta. Effectively, a Taylor series approximation to fn(t+delta), in delta, with basis given by the filter elements. If fn is None, it assumes fn=IRF[0], that is the first filter.

        >>> from numpy.random import *
        >>> from BrainSTAT.fMRIstat import HRF
        >>> from pylab import *
        >>> from numpy import *
        >>>
        >>> ddelta = 0.25
        >>> delta = arange(-4.5,4.5+ddelta, ddelta)
        >>> time = arange(0,20,0.2)
        >>>
        >>> hrf = HRF.HRF(deriv=True)
        >>>
        >>> canonical = HRF.canonical
        >>> taylor = hrf.deltaPCA(delta)
        >>> curplot = plot(time, taylor.components[1](time))
        >>> curplot = plot(time, taylor.components[0](time))
        >>> curtitle=title('Shift using Taylor series -- components')
        >>> show()
        >>>
        >>> curplot = plot(delta, taylor.coef[1](delta))
        >>> curplot = plot(delta, taylor.coef[0](delta))
        >>> curtitle = title('Shift using Taylor series -- coefficients')
        >>> show()
        >>>
        >>> curplot = plot(delta, taylor.inverse(delta))
        >>> curplot = plot(taylor.coef[1](delta) / taylor.coef[0](delta), delta)
        >>> curtitle = title('Shift using Taylor series -- inverting w1/w0')
        >>> show()


        '''
        if self.n != 2:
            raise ValueError, "Taylor series approximation assumes IRF has two components, IRF[0] -- a canonical IRF and IRF[1], its derivative."

        if not spectral: # use Taylor series approximation
            if dt is None:
                dt = self.dt
            time = arange(lower, tmax, dt)
            ntime = time.shape[0]

            if fn is None:
                fn = self.IRF[0]

            W = []
            H = []

            for i in range(delta.shape[0]):
                H.append(fn(time - delta[i]))
            H = array(H)

            if self.n >= 2:
                for _IRF in self.IRF:
                    W.append(_IRF(time))
                W = array(W)
            else:
                W = self.IRF(time)
                W.shape = (W.shape[0], 1)

            W = transpose(W)
            WH = dot(generalized_inverse(W), transpose(H))

            coef = []
            for i in range(self.n):
                coef.append(interpolant(delta, WH[i]))
            
            def approx(time, delta):
                value = 0
                for i in range(self.n):
                    value = value + coef[i](delta) * self.IRF[i](time)
                return value

            approx.coef = coef
            approx.components = self.IRF
            
            approx.theta, approx.inverse, approx.dinverse, approx.forward, approx.dforward = invertR(delta, approx.coef)
        
            return approx
        else:
            return deltaPCAsvd(self.IRF[0], delta, dt=dt, tmax=tmax, lower=lower, ncomp=2)


class GammaDENS:
    '''A class for a Gamma density -- only used so it knows how to differentiate itself.'''
    def __init__(self, alpha, nu, coef=1.0):
        self.alpha = alpha
        self.nu = nu
        self.coef = 1.0

    def __str__(self):
        return 'GammaDENS:alpha:%f,nu:%f,coef:%f' % (self.alpha, self.nu, self.coef)

    def __repr__(self):
        return self.__str__()

    def __mul__(self, const):
        self.coef = self.coef * const
        return self
    
    def __call__(self, x):
        '''Evaluate the Gamma density.'''
        _x = x * greater_equal(x, 0)
        return self.coef * _x**(self.alpha-1.) * exp(-self.nu*_x)

    def deriv(self):
        '''Differentiate a Gamma density. Returns a GammaCOMB that can evaluate the derivative.'''
        return GammaCOMB([[self.coef*(self.alpha-1), GammaDENS(self.alpha-1., self.nu)], [-self.coef*self.nu, GammaDENS(self.alpha, self.nu)]])

class GammaCOMB:
    def __init__(self, fns):
        self.fns = fns

    def __mul__(self, const):
        fns = []
        for fn in self.fns:
            fn[1] = fn[1] * const
            fns.append(fn)
        return GammaCOMB(fns)

    def __add__(self, other):
        fns = self.fns + other.fns
        return GammaCOMB(fns)

    def __call__(self, x):
        value = 0
        for coef, fn in self.fns:
            value = value + coef * fn(x)
        return value

    def deriv(self):
        fns = []
        for coef, fn in self.fns:
            comb = fn.deriv()
            comb.fns[0][0] = comb.fns[0][0] * coef
            comb.fns[1][0] = comb.fns[1][0] * coef
            fns = fns + comb.fns
        return GammaCOMB(fns)
    
class GammaHRF(Filter):
    '''A class that represents the Gamma basis in SPM: i.e. the filter is a collection of a certain number of Gamma densities. Parameters are specified as a kx2 matrix for k Gamma functions.

    >>> from BrainSTAT.fMRIstat import Filter
    >>> from pylab import *
    >>> from numpy import *
    >>> time = arange(0,50,0.2)
    >>> parameters = array([[9.6, 0.5], [5.0, 0.4], [10.0, 1.0]])
    >>> IRF = Filter.GammaHRF(parameters)
    >>> IRF.plot(time=time)
    >>> ylab = ylabel('Filters')
    >>> xlab = xlabel('Time (s)')
    >>> show()

    '''

    def __init__(self, parameters):
        fns = []
        for alpha, nu in parameters:
            fns.append(GammaDENS(alpha, nu))
        Filter.__init__(self, fns)

    def deriv(self):
        fns = []
        for fn in self.IRF:
            fns.append(fn.deriv())
        return fns
    
class FIR(Filter):
    '''A class for FIR filters: i.e. the filter is a collection of square waves. Parameters (start and duration) are specified as a kx2 matrix for k square waves.
    >>> from BrainSTAT.fMRIstat import Filter
    >>> from pylab import *
    >>> from numpy import *
    >>> parameters = array([[1., 2.], [2., 5.], [4., 8.]])
    >>> IRF = Filter.FIR(parameters)
    >>> IRF.plot(linestyle='steps')
    >>> ylab = ylabel('Filters')
    >>> xlab = xlabel('Time (s)')
    >>> show()

    '''

    def __init__(self, parameters):
        fns = []
        for start, duration in parameters:
            fns.append(WaveFunction(start, duration, 1.0))
        Filter.__init__(self, fns)
      
def invertR(delta, IRF, niter=20, verbose=False):
    '''If IRF has 2 components (w0, w1) return an estimate of the inverse of r=w1/w0, as in Liao et al. (2002). Fits a simple arctan model to the ratio w1/w0.?

    

    '''

    R = IRF[1](delta) / IRF[0](delta)

    def f(x, theta):
        a, b, c = theta
        _x = x[:,0]
        return a * arctan(b * _x) + c

    def grad(x, theta):
        a, b, c = theta
        value = zeros((3, x.shape[0]), Float)
        _x = x[:,0]
        value[0] = arctan(b * _x)
        value[1] = a / (1. + (b * _x) ** 2) * _x
        value[2] = 1.
        return transpose(value)

    c = max((delta / (pi/2)).flat)
    n = delta.shape[0]
    delta0 = (delta[n/2] - delta[n/2-1])/(R[n/2] - R[n/2-1])
    if delta0 < 0:
        c = max((delta / (pi/2)).flat) * 1.2
    else:
        c = -max((delta / (pi/2)).flat) * 1.2

    from neuroimaging.statistics import nlsmodel
    R.shape = (R.shape[0], 1)
    model = nlsmodel.NLSModel(Y=delta,
                              design=R,
                              f=f,
                              grad=grad,
                              theta=array([c, 1./(c*delta0), 0.]),
                              niter=niter)

    for iteration in model:
        if verbose:
            print model.theta
        model.next()

    a, b, c = model.theta

    def _deltahat(r):
        return a * arctan(b * r) + c

    def _ddeltahat(r):
        return a * b / (1 + (b * r)**2) 

    def _deltahatinv(d):
        return tan((d - c) / a) / b

    def _ddeltahatinv(d):
        return 1. / (a * b * cos((d - c) / a)**2)

    for fn in [_deltahat, _ddeltahat, _deltahatinv, _ddeltahatinv]:
        setattr(fn, 'a', a)
        setattr(fn, 'b', b)
        setattr(fn, 'c', c)

    return model.theta, _deltahat, _ddeltahat, _deltahatinv, _ddeltahatinv

def deltaPCAsvd(fn, delta, dt=None, tmax=50., lower=-15.0, ncomp=2):
    '''
    Perform a PCA expansion of fn, shifted over the values in delta. Effectively, a Taylor series approximation to fn(t+delta), in delta, with basis given by the singular value decomposition of the matrix.

    >>> from BrainSTAT.fMRIstat.HRF import HRF
    >>> from BrainSTAT.Visualization.Pylab import multipleLinePlot
    >>> from pylab import *
    >>> from numpy import *
    >>> time = arange(0,24,0.2)
    >>> delta = arange(-4.,4.,0.1)
    >>>
    >>> IRF = HRF()
    >>> spectral = deltaPCAsvd(IRF.IRF, delta, ncomp=2)
    >>> multipleLinePlot(spectral.components, time)
    >>> curtitle = title('Shift using SVD -- components')
    >>> show()
    >>>
    >>> multipleLinePlot(spectral.coef, delta)
    >>> curtitle = title('Shift using SVD -- cooefficients')
    >>> show()
    >>>
    >>> curplot = plot(delta, spectral.inverse(delta))
    >>> curplot = plot(spectral.coef[1](delta) / spectral.coef[0](delta), delta)
    >>> curtitle = title('Shift using SVD -- inverting w1/w0')
    >>> show()

    '''

    if dt is None:
        dt = self.df
    time = arange(lower, tmax, dt)
    ntime = time.shape[0]

    W = []
    H = []

    for i in range(delta.shape[0]):
        H.append(fn(time - delta[i]))
    H = array(H)

    U, S, V = singular_value_decomposition(transpose(H))
    prcnt_var_spectral = sum(S[0:ncomp]**2) / sum(S**2) * 100

    sumU = sum(U[:,0])
    
    US = U[:,0:ncomp] / sumU
    WS = V[0:ncomp] * sumU

    coef = []
    basis = []
    for i in range(ncomp):
        WS[i] = WS[i] * S[i]
        coef.append(interpolant(delta, WS[i]))
        basis.append(interpolant(time, US[:,i]))

    def approx(time, delta):
        value = 0
        for i in range(ncomp):
            value = value + coef[i](delta) * basis[i](time)
        return array(value)

    approx.coef = coef
    approx.components = basis
    if ncomp == 2:
        approx.theta, approx.inverse, approx.dinverse, approx.forward, approx.dforward = invertR(delta, approx.coef)

    return approx

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
