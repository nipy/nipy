import numpy as np
from numpy.random import random_integers as randint

from nipy.testing import *

from nipy.algorithms.kernel_smooth import LinearFilter
from nipy.core.api import Image
from nipy.core.reference.coordinate_map import CoordinateMap, Affine

from nipy.algorithms.kernel_smooth import sigma2fwhm, fwhm2sigma

"""
# FIXME: Need to make an automated test for this!
class test_Kernel(TestCase):
    @dec.gui
    import pylab
    from nipy.ui.visualization.viewer import BoxViewer
    def test_smooth(self):
        rho = Image("rho.hdr", repository)
        smoother = LinearFilter(rho.grid)

        srho = smoother.smooth(rho)
        view = BoxViewer(rho)
        view.draw()

        sview = BoxViewer(srho)
        sview.m = view.m
        sview.M = view.M
        sview.draw()
        pylab.show()
"""


def test_sigma_fwhm():
    """
    ensure that fwhm2sigma and sigma2fwhm are inverses of each other        
    """
    fwhm = np.arange(1.0, 5.0, 0.1)
    sigma = np.arange(1.0, 5.0, 0.1)
    yield assert_true, np.allclose(sigma2fwhm(fwhm2sigma(fwhm)), fwhm)
    yield assert_true, np.allclose(fwhm2sigma(sigma2fwhm(sigma)), sigma)

@dec.knownfailure
@dec.slow
def test_kernel():
    """
    Verify the the convolution with a delta function
    gives the correct answer.

    """

    tol = 0.9999
    sdtol = 1.0e-8
    for x in range(6):
        shape = randint(30,60,(3,))
        ii, jj, kk = randint(11,17, (3,))

        coordmap = Affine.from_start_step('ijk', 'xyz', 
                                          randint(5,20,(3,))*0.25,
                                          randint(5,10,(3,))*0.5)

        signal = np.zeros(shape)
        signal[ii,jj,kk] = 1.
        signal = Image(signal, coordmap=coordmap)
    
        kernel = LinearFilter(coordmap, shape, 
                              fwhm=randint(50,100)/10.)
        ssignal = kernel.smooth(signal)
        ssignal = np.asarray(ssignal)
        ssignal[:] *= kernel.norms[kernel.normalization]

        I = np.indices(ssignal.shape)
        I.shape = (kernel.coordmap.ndim[0], np.product(shape))
        i, j, k = I[:,np.argmax(ssignal[:].flat)]

        yield assert_equal, (i,j,k), (ii,jj,kk)

        Z = kernel.coordmap(I) - kernel.coordmap([i,j,k])

        _k = kernel(Z)
        _k.shape = ssignal.shape

        yield assert_true, (np.corrcoef(_k[:].flat, ssignal[:].flat)[0,1] > tol)

        yield assert_true, ((_k[:] - ssignal[:]).std() < sdtol)
            
        def _indices(i,j,k,axis):
            I = np.zeros((3,20))
            I[0] += i
            I[1] += j
            I[2] += k
            I[axis] += np.arange(-10,10)
            return I

        vx = ssignal[i,j,(k-10):(k+10)]
        vvx = coordmap(_indices(i,j,k,2)) - coordmap([[i],[j],[k]])
        yield assert_true, (np.corrcoef(vx, kernel(vvx))[0,1] > tol)

        vy = ssignal[i,(j-10):(j+10),k]
        vvy = coordmap(_indices(i,j,k,1)) - coordmap([[i],[j],[k]])
        yield assert_true, (np.corrcoef(vy, kernel(vvy))[0,1] > tol)

        vz = ssignal[(i-10):(i+10),j,k]
        vvz = coordmap(_indices(i,j,k,0)) - coordmap([[i],[j],[k]])
        yield assert_true, (np.corrcoef(vz, kernel(vvz))[0,1] > tol)

