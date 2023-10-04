# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Test for smoothing with kernels """
import numpy as np
import pytest
from numpy.random import randint
from numpy.testing import assert_array_almost_equal
from transforms3d.taitbryan import euler2mat

from ... import load_image
from ...core.api import AffineTransform, Image, compose, drop_io_dim
from ...testing import anatfile, funcfile
from ..kernel_smooth import LinearFilter, fwhm2sigma, sigma2fwhm


def test_anat_smooth():
    anat = load_image(anatfile)
    smoother = LinearFilter(anat.coordmap, anat.shape)
    sanat = smoother.smooth(anat)
    assert sanat.shape == anat.shape
    assert sanat.coordmap == anat.coordmap
    assert not np.allclose(sanat.get_fdata(), anat.get_fdata())


def test_funny_coordmap():
    # 5x4 affine should also work, and give same answer as 4x4
    func = load_image(funcfile)
    cmap = func.coordmap
    # Give the affine a rotation
    aff = np.eye(5)
    aff[:3,:3] = euler2mat(0.3, 0.2, 0.1)
    cmap_rot = AffineTransform(cmap.function_range,
                               cmap.function_range,
                               aff)
    func_rot = Image(func.get_fdata(), compose(cmap_rot, cmap))
    func1 = func_rot[...,1] # 5x4 affine
    smoother = LinearFilter(func1.coordmap, func1.shape)
    sfunc1 = smoother.smooth(func1) # OK
    # And same as for 4x4 affine
    cmap3d = drop_io_dim(cmap, 't')
    func3d = Image(func1.get_fdata(), cmap3d)
    smoother = LinearFilter(func3d.coordmap, func3d.shape)
    sfunc3d = smoother.smooth(func3d)
    assert sfunc1.shape == sfunc3d.shape
    assert_array_almost_equal(sfunc1.get_fdata(), sfunc3d.get_fdata())
    # And same with no rotation
    func_fresh = func[...,1] # 5x4 affine, no rotation
    smoother = LinearFilter(func_fresh.coordmap, func_fresh.shape)
    sfunc_fresh = smoother.smooth(func_fresh)
    assert sfunc1.shape == sfunc_fresh.shape
    assert_array_almost_equal(sfunc1.get_fdata(), sfunc_fresh.get_fdata())


def test_func_smooth():
    func = load_image(funcfile)
    smoother = LinearFilter(func.coordmap, func.shape)
    # should work, but currently broken : sfunc = smoother.smooth(func)
    pytest.raises(NotImplementedError, smoother.smooth, func)


def test_sigma_fwhm():
    # ensure that fwhm2sigma and sigma2fwhm are inverses of each other
    fwhm = np.arange(1.0, 5.0, 0.1)
    sigma = np.arange(1.0, 5.0, 0.1)
    assert np.allclose(sigma2fwhm(fwhm2sigma(fwhm)), fwhm)
    assert np.allclose(fwhm2sigma(sigma2fwhm(sigma)), sigma)


def test_kernel():
    # Verify that convolution with a delta function gives the correct
    # answer.
    tol = 0.9999
    sdtol = 1.0e-8
    for x in range(6):
        shape = randint(30, 60 + 1, (3,))
        # pos of delta
        ii, jj, kk = randint(11, 17 + 1, (3,))
        # random affine coordmap (diagonal and translations)
        coordmap = AffineTransform.from_start_step(
            'ijk', 'xyz',
            randint(5, 20 + 1, (3,)) * 0.25,
            randint(5, 10 + 1, (3,)) * 0.5)
        # delta function in 3D array
        signal = np.zeros(shape)
        signal[ii,jj,kk] = 1.
        signal = Image(signal, coordmap=coordmap)
        # A filter with coordmap, shape matched to image
        kernel = LinearFilter(coordmap, shape,
                              fwhm=randint(50, 100 + 1) / 10.)
        # smoothed normalized 3D array
        ssignal = kernel.smooth(signal).get_fdata()
        ssignal[:] *= kernel.norms[kernel.normalization]
        # 3 points * signal.size array
        I = np.indices(ssignal.shape)
        I.shape = (kernel.coordmap.ndims[0], np.prod(shape))
        # location of maximum in smoothed array
        i, j, k = I[:, np.argmax(ssignal[:].flat)]
        # same place as we put it before smoothing?
        assert (i,j,k) == (ii,jj,kk)
        # get physical points position relative to position of delta
        Z = kernel.coordmap(I.T) - kernel.coordmap([i,j,k])
        _k = kernel(Z)
        _k.shape = ssignal.shape
        assert np.corrcoef(_k[:].flat, ssignal[:].flat)[0,1] > tol
        assert (_k[:] - ssignal[:]).std() < sdtol

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
        assert np.corrcoef(vx, kernel(vvx))[0,1] > tol
        vy = ssignal[i,(j-10):(j+10),k]
        vvy = coordmap(_indices(i,j,k,1)) - xformed_ijk
        assert np.corrcoef(vy, kernel(vvy))[0,1] > tol
        vz = ssignal[(i-10):(i+10),j,k]
        vvz = coordmap(_indices(i,j,k,0)) - xformed_ijk
        assert np.corrcoef(vz, kernel(vvz))[0,1] > tol
