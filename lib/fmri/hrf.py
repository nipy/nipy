from numpy import *
from numpy.linalg import generalized_inverse, singular_value_decomposition
import filters
import enthought.traits as traits

def glover2GammaDENS(peak_hrf, fwhm_hrf):
    alpha = pow(peak_hrf / fwhm_hrf, 2) * 8 * log(2.0)
    beta = pow(fwhm_hrf, 2) / peak_hrf / 8 / log(2.0)
    coef = peak_hrf**(-alpha) * exp(peak_hrf / beta)
    return filters.GammaDENS(alpha + 1., 1. / beta) * coef

def _glover(peak_hrf=[5.4, 10.8], fwhm_hrf=[5.2, 7.35], dip=0.35):
    gamma1 = glover2GammaDENS(peak_hrf[0], fwhm_hrf[0])
    gamma2 = glover2GammaDENS(peak_hrf[1], fwhm_hrf[1])
    return filters.GammaCOMB([[1.0,gamma1],[-dip, gamma2]])

glover = _glover()
afni = filters.GammaDENS(9.6, 1.0/0.547)

class HRF(filters.Filter,traits.HasTraits):
    '''Use canonical Glover HRF as a filter.

    >>> from BrainSTAT.fMRIstat.HRF import HRF
    >>> from pylab import *
    >>> from numpy import *
    >>>
    >>> # Plotting the HRF
    ... time = arange(0,24,0.2)
    >>> IRF = HRF(deriv=True)
    >>> IRF.plot()
    >>> ylab = ylabel('Filters')
    >>> xlab = xlabel('Time (s)')
    >>> show()

    '''

    names = traits.ListStr(['glover'])
    
    def __init__(self, IRF=glover, deriv=False, delta=filters.Filter.delta):
        if deriv:
            dIRF = IRF.deriv()
            filters.Filter.__init__(self, [IRF, dIRF])
            self.delay = self.deltaPCA(delta, svd=True)
            self.names = ['glover', 'dglover']
        else:
            filters.Filter.__init__(self, IRF)
            
    def deriv(self):
        return self.IRF.deriv()

canonical = HRF()

class spectralHRF(filters.Filter):
    '''Use canonical Glover HRF as a filter.

    >>> from BrainSTAT.fMRIstat.HRF import spectralHRF
    >>> from pylab import *
    >>> from numpy import *
    >>>
    >>> # Plotting the HRF
    ... time = arange(0,24,0.2)
    >>> IRF = spectralHRF(ncomp=3)
    >>> IRF.plot()
    >>> ylab = ylabel('Filters')
    >>> xlab = xlabel('Time (s)')
    >>> show()

    '''
    def __init__(self, ncomp=2, reference=canonical, delta=filters.Filter.delta):
      
        from filters import deltaPCAsvd
        self.delay = deltaPCAsvd(reference, delta, ncomp=2)            
        filters.Filter.__init__(self, self.delay.components)
 
def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
    
