import numpy as N
import numpy.random as R
from numpy.testing import NumpyTest, NumpyTestCase

from neuroimaging.algorithms.kernel_smooth import LinearFilter
from neuroimaging.core.api import Image
from neuroimaging.utils.tests.data import repository
from neuroimaging.core.reference.grid import SamplingGrid, space
from neuroimaging.utils.test_decorators import gui
from neuroimaging.algorithms.kernel_smooth import sigma2fwhm, fwhm2sigma

from neuroimaging.defines import pylab_def


PYLAB_DEF, pylab = pylab_def()
if PYLAB_DEF:
    from neuroimaging.ui.visualization.viewer import BoxViewer
    from neuroimaging.modalities.fmri.pca import PCAmontage

class test_Kernel(NumpyTestCase):
    @gui
    def test_smooth(self):
        rho = Image("rho.hdr", repository)
        smoother = LinearFilter(rho.grid)

        if PYLAB_DEF:
            srho = smoother.smooth(rho)
            view = BoxViewer(rho)
            view.draw()

            sview = BoxViewer(srho)
            sview.m = view.m
            sview.M = view.M
            sview.draw()
            pylab.show()

class test_SigmaFWHM(NumpyTestCase):
    def test_sigma_fwhm(self):
        """
        ensure that fwhm2sigma and sigma2fwhm are inverses of each other        
        """
        fwhm = N.arange(1.0, 5.0, 0.1)
        sigma = N.arange(1.0, 5.0, 0.1)
        N.testing.assert_almost_equal(sigma2fwhm(fwhm2sigma(fwhm)), fwhm)
        N.testing.assert_almost_equal(fwhm2sigma(sigma2fwhm(sigma)), sigma)

    def test_kernel(self):
        """
        Verify the the convolution with a delta function
        gives the correct answer.

        """

        tol = 0.9999
        sdtol = 1.0e-8
        for i in range(6):
            shape = R.random_integers(30,60,(3,))
            ii, jj, kk = R.random_integers(11,17, (3,))

            grid = SamplingGrid.from_start_step(names=space, shape=shape, step=R.random_integers(5,10,(3,))*0.5, start=R.random_integers(5,20,(3,))*0.25)

            signal = N.zeros(shape)
            signal[ii,jj,kk] = 1.
            signal = Image(signal, grid=grid)
    
            kernel = LinearFilter(grid, fwhm=R.random_integers(50,100)/10.)
            ssignal = kernel.smooth(signal)
            ssignal[:] *= kernel.norms[kernel.normalization]

            I = N.indices(ssignal.shape)
            I.shape = (3, N.product(shape))
            i, j, k = I[:,N.argmax(ssignal[:].flat)]

            self.assertTrue((i,j,k) == (ii,jj,kk))

            II = N.copy(I)
            Z = kernel.grid.mapping(I) - kernel.grid.mapping([[i],[j],[k]])

            _k = kernel(Z)
            _k.shape = ssignal.shape
            self.assertTrue(N.corrcoef(_k[:].flat, ssignal[:].flat)[0,1] > tol)
            self.assertTrue((_k[:] - ssignal[:]).std() < sdtol)
            
            def _indices(i,j,k,axis):
                I = N.zeros((3,20))
                I[0] += i
                I[1] += j
                I[2] += k
                I[axis] += N.arange(-10,10)
                return I

            vx = ssignal[i,j,(k-10):(k+10)]
            vvx = grid.mapping(_indices(i,j,k,2)) - grid.mapping([[i],[j],[k]])
            self.assertTrue(N.corrcoef(vx, kernel(vvx))[0,1] > tol)

            vy = ssignal[i,(j-10):(j+10),k]
            vvy = grid.mapping(_indices(i,j,k,1)) - grid.mapping([[i],[j],[k]])
            self.assertTrue(N.corrcoef(vy, kernel(vvy))[0,1] > tol)

            vz = ssignal[(i-10):(i+10),j,k]
            vvz = grid.mapping(_indices(i,j,k,0)) - grid.mapping([[i],[j],[k]])
            self.assertTrue(N.corrcoef(vz, kernel(vvz))[0,1] > tol)


from neuroimaging.utils.testutils import make_doctest_suite
test_suite = make_doctest_suite('neuroimaging.algorithms.kernel_smooth')

        
if __name__ == '__main__':
    NumpyTest.run()
