# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
from numpy.random import random_integers as randint

from nipy import load_image
from nipy.algorithms.kernel_smooth import LinearFilter
from nipy.core.api import Image
from nipy.core.reference.coordinate_map import AffineTransform

from nipy.algorithms.kernel_smooth import sigma2fwhm, fwhm2sigma

from nipy.testing import (assert_true, assert_equal, assert_raises,
                          dec, parametric,
                          anatfile, funcfile)

@parametric
def test_anat_smooth():
    anat = load_image(anatfile)
    smoother = LinearFilter(anat.coordmap, anat.shape)
    sanat = smoother.smooth(anat)


@parametric
def test_func_smooth():
    func = load_image(funcfile)
    smoother = LinearFilter(func.coordmap, func.shape)
    # should work, but currently broken : sfunc = smoother.smooth(func)
    yield assert_raises(NotImplementedError, smoother.smooth, func)


@parametric
def test_sigma_fwhm():
    # ensure that fwhm2sigma and sigma2fwhm are inverses of each other        
    fwhm = np.arange(1.0, 5.0, 0.1)
    sigma = np.arange(1.0, 5.0, 0.1)
    yield assert_true(np.allclose(sigma2fwhm(fwhm2sigma(fwhm)), fwhm))
    yield assert_true(np.allclose(fwhm2sigma(sigma2fwhm(sigma)), sigma))


@parametric
def test_kernel():
    # Verify that convolution with a delta function gives the correct
    # answer.
    tol = 0.9999
    sdtol = 1.0e-8
    for x in range(6):
        shape = randint(30,60,(3,))
        ii, jj, kk = randint(11,17, (3,))
        coordmap = AffineTransform.from_start_step('ijk', 'xyz', 
                                          randint(5,20,(3,))*0.25,
                                          randint(5,10,(3,))*0.5)
        # delta function in 3D array
        signal = np.zeros(shape)
        signal[ii,jj,kk] = 1.
        signal = Image(signal, coordmap=coordmap)
        kernel = LinearFilter(coordmap, shape, 
                              fwhm=randint(50,100)/10.)
        # smoothed normalized 3D array
        ssignal = kernel.smooth(signal)
        ssignal = np.asarray(ssignal)
        ssignal[:] *= kernel.norms[kernel.normalization]
        # ssignal.size x 3 points array
        I = np.indices(ssignal.shape)
        I.shape = (kernel.coordmap.ndims[0], np.product(shape))
        I = I.T
        # location of maximum in smoothed array
        i, j, k = I[np.argmax(ssignal[:].flat),:]
        # same place as we put it before smoothing?
        yield assert_equal((i,j,k), (ii,jj,kk))
        # get physical points relative to position of delta
        Z = kernel.coordmap(I) - kernel.coordmap([i,j,k])
        _k = kernel(Z)
        _k.shape = ssignal.shape
        yield assert_true((np.corrcoef(_k[:].flat, ssignal[:].flat)[0,1] > tol))
        yield assert_true(((_k[:] - ssignal[:]).std() < sdtol))
            
        def _indices(i,j,k,axis):
            I = np.zeros((3,20))
            I[0] += i
            I[1] += j
            I[2] += k
            I[axis] += np.arange(-10,10)
            return I.T

        vx = ssignal[i,j,(k-10):(k+10)]
        xformed_ijk = coordmap([i, j, k])
        vvx = coordmap(_indices(i,j,k,2)) - xformed_ijk
        yield assert_true((np.corrcoef(vx, kernel(vvx))[0,1] > tol))
        vy = ssignal[i,(j-10):(j+10),k]
        vvy = coordmap(_indices(i,j,k,1)) - xformed_ijk
        yield assert_true((np.corrcoef(vy, kernel(vvy))[0,1] > tol))
        vz = ssignal[(i-10):(i+10),j,k]
        vvz = coordmap(_indices(i,j,k,0)) - xformed_ijk
        yield assert_true((np.corrcoef(vz, kernel(vvz))[0,1] > tol))

