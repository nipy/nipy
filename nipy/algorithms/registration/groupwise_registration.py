# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Motion correction / motion correction with slice timing

Routines implementing motion correction and motion correction combined with
slice-timing.

See:

Roche, Alexis (2011) A four-dimensional registration algorithm with application
to joint correction of motion and slice timing in fMRI. *Medical Imaging, IEEE
Transactions on*;  30:1546--1554
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import warnings

import numpy as np

from ...externals.six import string_types

from nibabel.affines import apply_affine

from ...fixes.nibabel import io_orientation
from ...io.nibcompat import get_header
from ...core.image.image_spaces import (make_xyz_image,
                                        xyz_affine,
                                        as_xyz_image)
from ..slicetiming import timefuncs
from .affine import Rigid, Affine
from .optimizer import configure_optimizer, use_derivatives
from .type_check import (check_type, check_type_and_shape)
from ._registration import (_cspline_transform,
                            _cspline_sample3d,
                            _cspline_sample4d)


VERBOSE = os.environ.get('NIPY_DEBUG_PRINT', False)
INTERLEAVED = None
XTOL = 1e-5
FTOL = 1e-5
GTOL = 1e-5
STEPSIZE = 1e-6
SMALL = 1e-20
MAXITER = 64
MAXFUN = None


def interp_slice_times(Z, slice_times, tr):
    Z = np.asarray(Z)
    nslices = len(slice_times)
    aux = np.asarray(list(slice_times) + [slice_times[0] + tr])
    Zf = np.floor(Z).astype('int')
    w = Z - Zf
    Zal = Zf % nslices
    Za = Zal + w
    ret = (1 - w) * aux[Zal] + w * aux[Zal + 1]
    ret += (Z - Za)
    return ret


def scanner_coords(xyz, affine, from_world, to_world):
    Tv = np.dot(from_world, np.dot(affine, to_world))
    XYZ = apply_affine(Tv, xyz)
    return XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]


def make_grid(dims, subsampling=(1, 1, 1), borders=(0, 0, 0)):
    slices = [slice(b, d - b, s)\
                  for d, s, b in zip(dims, subsampling, borders)]
    xyz = np.mgrid[slices]
    xyz = np.rollaxis(xyz, 0, 4)
    xyz = np.reshape(xyz, [np.prod(xyz.shape[0:-1]), 3])
    return xyz


def guess_slice_axis_and_direction(slice_info, affine):
    if slice_info is None:
        orient = io_orientation(affine)
        slice_axis = int(np.where(orient[:, 0] == 2)[0])
        slice_direction = int(orient[slice_axis, 1])
    else:
        slice_axis = int(slice_info[0])
        slice_direction = int(slice_info[1])
    return slice_axis, slice_direction

def tr_from_header(images):
    """ Return the TR from the header of an image or list of images.

    Parameters
    ----------
    images : image or list of images
      Single or multiple input 4d images representing one or
      several sessions.

    Returns
    -------
    float
      Repetition time, as specified in NIfTI header.

    Raises
    ------
    ValueError
      if the TR between the images is inconsistent.
    """
    if not isinstance(images, list):
        images = [images]
    images_tr = None
    for image in images:
        tr = get_header(image).get_zooms()[3]
        if images_tr is None:
            images_tr = tr
        if tr != images_tr:
            raise ValueError('TR inconsistent between images.')
    return images_tr

class Image4d(object):
    """
    Class to represent a sequence of 3d scans (possibly acquired on a
    slice-by-slice basis).

    Object remains empty until the data array is actually loaded in memory.

    Parameters
    ----------
      data : nd array or proxy (function that actually gets the array)
    """
    def __init__(self, data, affine, tr, slice_times, slice_info=None):
        """
        Configure fMRI acquisition time parameters.
        """
        self.affine = np.asarray(affine)
        self.tr = float(tr)

        # guess the slice axis and direction (z-axis)
        self.slice_axis, self.slice_direction =\
            guess_slice_axis_and_direction(slice_info, self.affine)

        # unformatted parameters
        self._slice_times = slice_times

        if isinstance(data, np.ndarray):
            self._data = data
            self._shape = data.shape
            self._get_data = None
            self._init_timing_parameters()
        else:
            self._data = None
            self._shape = None
            self._get_data = data

    def _load_data(self):
        self._data = self._get_data()
        self._shape = self._data.shape
        self._init_timing_parameters()

    def get_data(self):
        if self._data is None:
            self._load_data()
        return self._data

    def get_shape(self):
        if self._shape is None:
            self._load_data()
        return self._shape

    def _init_timing_parameters(self):
        # Number of slices
        nslices = self.get_shape()[self.slice_axis]
        self.nslices = nslices
        # Set slice times
        if isinstance(self._slice_times, (int, float)):
            # If a single value is provided, assume synchronous slices
            self.slice_times = np.zeros(nslices)
            self.slice_times.fill(self._slice_times)
        else:
            # Verify correctness of provided slice times
            if not len(self._slice_times) == nslices:
                raise ValueError(
                    "Incorrect slice times were provided. There are %d "
                    "slices in the volume, `slice_times` argument has length %d"
                    % (nslices, len(self._slice_times)))
            self.slice_times = np.asarray(self._slice_times)
        # Check that slice times are smaller than repetition time
        if np.max(self.slice_times) > self.tr:
            raise ValueError("slice times should be smaller than repetition time")

    def z_to_slice(self, z):
        """
        Account for the fact that slices may be stored in reverse
        order wrt the scanner coordinate system convention (slice 0 ==
        bottom of the head)
        """
        if self.slice_direction < 0:
            return self.nslices - 1 - z
        else:
            return z

    def scanner_time(self, zv, t):
        """
        tv = scanner_time(zv, t)
        zv, tv are grid coordinates; t is an actual time value.
        """
        corr = interp_slice_times(self.z_to_slice(zv),
                                  self.slice_times,
                                  self.tr)
        return (t - corr) / self.tr

    def free_data(self):
        if self._get_data is not None:
            self._data = None



class Realign4dAlgorithm(object):

    def __init__(self,
                 im4d,
                 affine_class=Rigid,
                 transforms=None,
                 time_interp=True,
                 subsampling=(1, 1, 1),
                 refscan=0,
                 borders=(1, 1, 1),
                 optimizer='ncg',
                 optimize_template=True,
                 xtol=XTOL,
                 ftol=FTOL,
                 gtol=GTOL,
                 stepsize=STEPSIZE,
                 maxiter=MAXITER,
                 maxfun=MAXFUN):

        # Check arguments
        check_type_and_shape(subsampling, int, 3)
        check_type(refscan, int, accept_none=True)
        check_type_and_shape(borders, int, 3)
        check_type(xtol, float)
        check_type(ftol, float)
        check_type(gtol, float)
        check_type(stepsize, float)
        check_type(maxiter, int)
        check_type(maxfun, int, accept_none=True)

        # Get dimensional parameters
        self.dims = im4d.get_shape()
        self.nscans = self.dims[3]
        # Reduce borders if spatial image dimension too small to avoid
        # getting an empty volume of interest
        borders = [min(b, d/2 - (not d%2)) for (b, d) in zip(borders, self.dims[0:3])]
        self.xyz = make_grid(self.dims[0:3], subsampling, borders)
        masksize = self.xyz.shape[0]
        self.data = np.zeros([masksize, self.nscans], dtype='double')

        # Initialize space/time transformation parameters
        self.affine = im4d.affine
        self.inv_affine = np.linalg.inv(self.affine)
        if transforms is None:
            self.transforms = [affine_class() for scan in range(self.nscans)]
        else:
            self.transforms = transforms

        # Compute the 4d cubic spline transform
        self.time_interp = time_interp
        if time_interp:
            self.timestamps = im4d.tr * np.arange(self.nscans)
            self.scanner_time = im4d.scanner_time
            self.cbspline = _cspline_transform(im4d.get_data())
        else:
            self.cbspline = np.zeros(self.dims, dtype='double')
            for t in range(self.dims[3]):
                self.cbspline[:, :, :, t] =\
                    _cspline_transform(im4d.get_data()[:, :, :, t])

        # The reference scan conventionally defines the head
        # coordinate system
        self.optimize_template = optimize_template
        if not optimize_template and refscan is None:
            self.refscan = 0
        else:
            self.refscan = refscan

        # Set the minimization method
        self.set_fmin(optimizer, stepsize,
                      xtol=xtol,
                      ftol=ftol,
                      gtol=gtol,
                      maxiter=maxiter,
                      maxfun=maxfun)

        # Auxiliary array for realignment estimation
        self._res = np.zeros(masksize, dtype='double')
        self._res0 = np.zeros(masksize, dtype='double')
        self._aux = np.zeros(masksize, dtype='double')
        self.A = np.zeros((masksize, self.transforms[0].param.size),
                          dtype='double')
        self._pc = None

    def resample(self, t):
        """
        Resample a particular time frame on the (sub-sampled) working
        grid.

        x,y,z,t are "head" grid coordinates
        X,Y,Z,T are "scanner" grid coordinates
        """
        X, Y, Z = scanner_coords(self.xyz, self.transforms[t].as_affine(),
                                 self.inv_affine, self.affine)
        if self.time_interp:
            T = self.scanner_time(Z, self.timestamps[t])
            _cspline_sample4d(self.data[:, t],
                              self.cbspline,
                              X, Y, Z, T,
                              mx='reflect',
                              my='reflect',
                              mz='reflect',
                              mt='reflect')
        else:
            _cspline_sample3d(self.data[:, t],
                              self.cbspline[:, :, :, t],
                              X, Y, Z,
                              mx='reflect',
                              my='reflect',
                              mz='reflect')

    def resample_full_data(self):
        if VERBOSE:
            print('Gridding...')
        xyz = make_grid(self.dims[0:3])
        res = np.zeros(self.dims)
        for t in range(self.nscans):
            if VERBOSE:
                print('Fully resampling scan %d/%d' % (t + 1, self.nscans))
            X, Y, Z = scanner_coords(xyz, self.transforms[t].as_affine(),
                                     self.inv_affine, self.affine)
            if self.time_interp:
                T = self.scanner_time(Z, self.timestamps[t])
                _cspline_sample4d(res[:, :, :, t],
                                  self.cbspline,
                                  X, Y, Z, T,
                                  mt='nearest')
            else:
                _cspline_sample3d(res[:, :, :, t],
                                  self.cbspline[:, :, :, t],
                                  X, Y, Z)
        return res

    def set_fmin(self, optimizer, stepsize, **kwargs):
        """
        Return the minimization function
        """
        self.stepsize = stepsize
        self.optimizer = optimizer
        self.optimizer_kwargs = kwargs
        self.optimizer_kwargs.setdefault('xtol', XTOL)
        self.optimizer_kwargs.setdefault('ftol', FTOL)
        self.optimizer_kwargs.setdefault('gtol', GTOL)
        self.optimizer_kwargs.setdefault('maxiter', MAXITER)
        self.optimizer_kwargs.setdefault('maxfun', MAXFUN)
        self.use_derivatives = use_derivatives(self.optimizer)

    def init_instant_motion(self, t):
        """
        Pre-compute and cache some constants (at fixed time) for
        repeated computations of the alignment energy.

        The idea is to decompose the average temporal variance via:

        V = (n-1)/n V* + (n-1)/n^2 (x-m*)^2

        with x the considered volume at time t, and m* the mean of all
        resampled volumes but x. Only the second term is variable when

        one volumes while the others are fixed. A similar
        decomposition is used for the global variance, so we end up
        with:

        V/V0 = [nV* + (x-m*)^2] / [nV0* + (x-m0*)^2]
        """
        fixed = list(range(self.nscans))
        fixed.remove(t)
        aux = self.data[:, fixed]
        if self.optimize_template:
            self.mu = np.mean(aux, 1)
        self.offset = self.nscans * np.mean((aux.T - self.mu) ** 2)
        self.mu0 = np.mean(aux)
        self.offset0 = self.nscans * np.mean((aux - self.mu0) ** 2)
        self._t = t
        self._pc = None

    def set_transform(self, t, pc):
        self.transforms[t].param = pc
        self.resample(t)

    def _init_energy(self, pc):
        if pc is self._pc:
            return
        self.set_transform(self._t, pc)
        self._pc = pc
        self._res[:] = self.data[:, self._t] - self.mu[:]
        self._V = np.maximum(self.offset + np.mean(self._res ** 2), SMALL)
        self._res0[:] = self.data[:, self._t] - self.mu0
        self._V0 = np.maximum(self.offset0 + np.mean(self._res0 ** 2), SMALL)

        if self.use_derivatives:
            # linearize the data wrt the transform parameters
            # use the auxiliary array to save the current resampled data
            self._aux[:] = self.data[:, self._t]
            basis = np.eye(6)
            for j in range(pc.size):
                self.set_transform(self._t, pc + self.stepsize * basis[j])
                self.A[:, j] = (self.data[:, self._t] - self._aux)\
                    / self.stepsize
            self.transforms[self._t].param = pc
            self.data[:, self._t] = self._aux[:]
            # pre-compute gradient and hessian of numerator and
            # denominator
            c = 2 / float(self.data.shape[0])
            self._dV = c * np.dot(self.A.T, self._res)
            self._dV0 = c * np.dot(self.A.T, self._res0)
            self._H = c * np.dot(self.A.T, self.A)

    def _energy(self):
        """
        The alignment energy is defined as the log-ratio between the
        average temporal variance in the sequence and the global
        spatio-temporal variance.
        """
        return np.log(self._V / self._V0)

    def _energy_gradient(self):
        return self._dV / self._V - self._dV0 / self._V0

    def _energy_hessian(self):
        return (1 / self._V - 1 / self._V0) * self._H\
            - np.dot(self._dV, self._dV.T) / np.maximum(self._V ** 2, SMALL)\
            + np.dot(self._dV0, self._dV0.T) / np.maximum(self._V0 ** 2, SMALL)

    def estimate_instant_motion(self, t):
        """
        Estimate motion parameters at a particular time.
        """
        if VERBOSE:
            print('Estimating motion at time frame %d/%d...'
                  % (t + 1, self.nscans))

        def f(pc):
            self._init_energy(pc)
            return self._energy()

        def fprime(pc):
            self._init_energy(pc)
            return self._energy_gradient()

        def fhess(pc):
            self._init_energy(pc)
            return self._energy_hessian()

        self.init_instant_motion(t)
        fmin, args, kwargs =\
            configure_optimizer(self.optimizer,
                                fprime=fprime,
                                fhess=fhess,
                                **self.optimizer_kwargs)

        # With scipy >= 0.9, some scipy minimization functions like
        # fmin_bfgs may crash due to the subroutine
        # `scalar_search_armijo` returning None as a stepsize when
        # unhappy about the objective function. This seems to have the
        # potential to occur in groupwise registration when using
        # strong image subsampling, i.e. at the coarser levels of the
        # multiscale pyramid. To avoid crashes, we insert a try/catch
        # instruction.
        try:
            pc = fmin(f, self.transforms[t].param, disp=VERBOSE,
                      *args, **kwargs)
            self.set_transform(t, pc)
        except:
            warnings.warn('Minimization failed')

    def estimate_motion(self):
        """
        Optimize motion parameters for the whole sequence. All the
        time frames are initially resampled according to the current
        space/time transformation, the parameters of which are further
        optimized sequentially.
        """
        for t in range(self.nscans):
            if VERBOSE:
                print('Resampling scan %d/%d' % (t + 1, self.nscans))
            self.resample(t)

        # Set the template as the reference scan (will be overwritten
        # if template is to be optimized)
        if not hasattr(self, 'template'):
            self.mu = self.data[:, self.refscan].copy()
        for t in range(self.nscans):
            self.estimate_instant_motion(t)
            if VERBOSE:
                print(self.transforms[t])

    def align_to_refscan(self):
        """
        The `motion_estimate` method aligns scans with an online
        template so that spatial transforms map some average head
        space to the scanner space. To conventionally redefine the
        head space as being aligned with some reference scan, we need
        to right compose each head_average-to-scanner transform with
        the refscan's 'to head_average' transform.
        """
        if self.refscan is None:
            return
        Tref_inv = self.transforms[self.refscan].inv()
        for t in range(self.nscans):
            self.transforms[t] = (self.transforms[t]).compose(Tref_inv)


def resample4d(im4d, transforms, time_interp=True):
    """
    Resample a 4D image according to the specified sequence of spatial
    transforms, using either 4D interpolation if `time_interp` is True
    and 3D interpolation otherwise.
    """
    r = Realign4dAlgorithm(im4d, transforms=transforms,
                           time_interp=time_interp)
    res = r.resample_full_data()
    im4d.free_data()
    return res


def adjust_subsampling(speedup, dims):
    dims = np.array(dims)
    aux = np.maximum(speedup * dims / np.prod(dims) ** (1 / 3.), [1, 1, 1])
    return aux.astype('int')


def single_run_realign4d(im4d,
                         affine_class=Rigid,
                         time_interp=True,
                         loops=5,
                         speedup=5,
                         refscan=0,
                         borders=(1, 1, 1),
                         optimizer='ncg',
                         xtol=XTOL,
                         ftol=FTOL,
                         gtol=GTOL,
                         stepsize=STEPSIZE,
                         maxiter=MAXITER,
                         maxfun=MAXFUN):
    """
    Realign a single run in space and time.

    Parameters
    ----------
    im4d : Image4d instance

    speedup : int or sequence
      If a sequence, implement a multi-scale realignment
    """
    if not type(loops) in (list, tuple, np.array):
        loops = [loops]
    repeats = len(loops)

    def format_arg(x):
        if not type(x) in (list, tuple, np.array):
            x = [x for i in range(repeats)]
        else:
            if not len(x) == repeats:
                raise ValueError('inconsistent length in arguments')
        return x

    speedup = format_arg(speedup)
    optimizer = format_arg(optimizer)
    xtol = format_arg(xtol)
    ftol = format_arg(ftol)
    gtol = format_arg(gtol)
    stepsize = format_arg(stepsize)
    maxiter = format_arg(maxiter)
    maxfun = format_arg(maxfun)

    transforms = None
    opt_params = zip(loops, speedup, optimizer,
                     xtol, ftol, gtol,
                     stepsize, maxiter, maxfun)

    for loops_, speedup_, optimizer_, xtol_, ftol_, gtol_,\
            stepsize_, maxiter_, maxfun_ in opt_params:
        subsampling = adjust_subsampling(speedup_, im4d.get_shape()[0:3])

        r = Realign4dAlgorithm(im4d,
                               transforms=transforms,
                               affine_class=affine_class,
                               time_interp=time_interp,
                               subsampling=subsampling,
                               refscan=refscan,
                               borders=borders,
                               optimizer=optimizer_,
                               xtol=xtol_,
                               ftol=ftol_,
                               gtol=gtol_,
                               stepsize=stepsize_,
                               maxiter=maxiter_,
                               maxfun=maxfun_)

        for loop in range(loops_):
            r.estimate_motion()

        r.align_to_refscan()
        transforms = r.transforms
        im4d.free_data()

    return transforms


def realign4d(runs,
              affine_class=Rigid,
              time_interp=True,
              align_runs=True,
              loops=5,
              between_loops=5,
              speedup=5,
              refscan=0,
              borders=(1, 1, 1),
              optimizer='ncg',
              xtol=XTOL,
              ftol=FTOL,
              gtol=GTOL,
              stepsize=STEPSIZE,
              maxiter=MAXITER,
              maxfun=MAXFUN):
    """
    Parameters
    ----------
    runs : list of Image4d objects

    Returns
    -------
    transforms : list
                 nested list of rigid transformations


    transforms map an 'ideal' 4d grid (conventionally aligned with the
    first scan of the first run) to the 'acquisition' 4d grid for each
    run
    """

    # Single-session case
    if not type(runs) in (list, tuple, np.array):
        runs = [runs]
    nruns = len(runs)
    if nruns == 1:
        align_runs = False

    # Correct motion and slice timing in each sequence separately
    transforms = [single_run_realign4d(run,
                                       affine_class=affine_class,
                                       time_interp=time_interp,
                                       loops=loops,
                                       speedup=speedup,
                                       refscan=refscan,
                                       borders=borders,
                                       optimizer=optimizer,
                                       xtol=xtol,
                                       ftol=ftol,
                                       gtol=gtol,
                                       stepsize=stepsize,
                                       maxiter=maxiter,
                                       maxfun=maxfun) for run in runs]

    if not align_runs:
        return transforms, transforms, None

    # Correct between-session motion using the mean image of each
    # corrected run, and creating a fake time series with no temporal
    # smoothness. If the runs have different affines, a correction is
    # applied to the transforms associated with each run (except for
    # the first run) so that all images included in the fake series
    # have the same affine, namely that of the first run.
    is_same_affine = lambda a1, a2: np.max(np.abs(a1 - a2)) < 1e-5
    mean_img_shape = list(runs[0].get_shape()[0:3]) + [nruns]
    mean_img_data = np.zeros(mean_img_shape)
    for i in range(nruns):
        if is_same_affine(runs[0].affine, runs[i].affine):
            transforms_i = transforms[i]
        else:
            runs[i].affine = runs[0].affine
            aff_corr = Affine(np.dot(runs[0].affine,
                                     np.linalg.inv(runs[i].affine)))
            transforms_i = [aff_corr.compose(Affine(t.as_affine()))\
                                for t in transforms[i]]
        corr_run = resample4d(runs[i], transforms=transforms_i,
                              time_interp=time_interp)
        mean_img_data[..., i] = corr_run.mean(3)
    del corr_run

    mean_img = Image4d(mean_img_data, affine=runs[0].affine,
                       tr=1.0, slice_times=0)
    transfo_mean = single_run_realign4d(mean_img,
                                        affine_class=affine_class,
                                        time_interp=False,
                                        loops=between_loops,
                                        speedup=speedup,
                                        borders=borders,
                                        optimizer=optimizer,
                                        xtol=xtol,
                                        ftol=ftol,
                                        gtol=gtol,
                                        stepsize=stepsize,
                                        maxiter=maxiter,
                                        maxfun=maxfun)

    # Compose transformations for each run
    ctransforms = [None for i in range(nruns)]
    for i in range(nruns):
        ctransforms[i] = [t.compose(transfo_mean[i]) for t in transforms[i]]
    return ctransforms, transforms, transfo_mean


class Realign4d(object):

    def __init__(self, images, tr, slice_times=None, slice_info=None,
                 affine_class=Rigid):
        """
        Spatiotemporal realignment class for series of 3D images.

        The algorithm performs simultaneous motion and slice timing
        correction for fMRI series or other data where slices are not
        acquired simultaneously.

        Parameters
        ----------
        images : image or list of images
          Single or multiple input 4d images representing one or
          several sessions.

        tr : float
          Inter-scan repetition time, i.e. the time elapsed between
          two consecutive scans. The unit in which `tr` is given is
          arbitrary although it needs to be consistent with the
          `slice_times` argument.

        slice_times : None or array-like
          If None, slices are assumed to be acquired simultaneously
          hence no slice timing correction is performed. If
          array-like, then the slice acquisition times.

        slice_info : None or tuple, optional
          None, or a tuple with slice axis as the first element and
          direction as the second, for instance (2, 1).  If None, then
          guess the slice axis, and direction, as the closest to the z
          axis, as estimated from the affine.
        """
        self._init(images, tr, slice_times, slice_info, affine_class)

    def _init(self, images, tr, slice_times, slice_info, affine_class):
        """
        Generic initialization method.
        """
        if slice_times is None:
            tr = 1.0
            slice_times = 0.0
            time_interp = False
        else:
            time_interp = True
        if not isinstance(images, (list, tuple, np.ndarray)):
            images = [images]
        if tr is None:
            raise ValueError('Repetition time cannot be None.')
        if tr == 0:
            raise ValueError('Repetition time cannot be zero.')
        self.affine_class = affine_class
        self.slice_times = slice_times
        self.tr = tr
        self._runs = []
        # Note that, the affine of each run may be different. This is
        # the case, for instance, if the subject exits the scanner
        # inbetween sessions.
        for im in images:
            xyz_img = as_xyz_image(im)
            self._runs.append(Image4d(xyz_img.get_data,
                                      xyz_affine(xyz_img),
                                      tr,
                                      slice_times=slice_times,
                                      slice_info=slice_info))
        self._transforms = [None for run in self._runs]
        self._within_run_transforms = [None for run in self._runs]
        self._mean_transforms = [None for run in self._runs]
        self._time_interp = time_interp

    def estimate(self,
                 loops=5,
                 between_loops=None,
                 align_runs=True,
                 speedup=5,
                 refscan=0,
                 borders=(1, 1, 1),
                 optimizer='ncg',
                 xtol=XTOL,
                 ftol=FTOL,
                 gtol=GTOL,
                 stepsize=STEPSIZE,
                 maxiter=MAXITER,
                 maxfun=MAXFUN):
        """Estimate motion parameters.

        Parameters
        ----------
        loops : int or sequence of ints
            Determines the number of iterations performed to realign
            scans within each run for each pass defined by the
            ``speedup`` argument. For instance, setting ``speedup`` ==
            (5,2) and ``loops`` == (5,1) means that 5 iterations are
            performed in a first pass where scans are subsampled by an
            isotropic factor 5, followed by one iteration where scans
            are subsampled by a factor 2.
        between_loops : None, int or sequence of ints
            Similar to ``loops`` for between-run motion
            estimation. Determines the number of iterations used to
            realign scans across runs, a procedure similar to
            within-run realignment that uses the mean images from each
            run. If None, assumed to be the same as ``loops``.
            The setting used in the experiments described in Roche,
            IEEE TMI 2011, was: ``speedup`` = (5, 2), ``loops`` = (5,
            1) and ``between_loops`` = (5, 1).
        align_runs : bool
            Determines whether between-run motion is estimated or
            not. If False, the ``between_loops`` argument is ignored.
        speedup: int or sequence of ints
            Determines an isotropic sub-sampling factor, or a sequence
            of such factors, applied to the scans to perform motion
            estimation. If a sequence, several estimation passes are
            applied.
        refscan : None or int
            Defines the number of the scan used as the reference
            coordinate system for each run. If None, a reference
            coordinate system is defined internally that does not
            correspond to any particular scan. Note that the
            coordinate system associated with the first run is always
        borders : sequence of ints
            Should be of length 3. Determines the field of view for
            motion estimation in terms of the number of slices at each
            extremity of the reference grid that are ignored for
            motion parameter estimation. For instance,
            ``borders``==(1,1,1) means that the realignment cost
            function will not take into account voxels located in the
            first and last axial/sagittal/coronal slices in the
            reference grid. Please note that this choice only affects
            parameter estimation but does not affect image resampling
            in any way, see ``resample`` method.
        optimizer : str
            Defines the optimization method. One of 'simplex',
            'powell', 'cg', 'ncg', 'bfgs' and 'steepest'.
        xtol : float
            Tolerance on variations of transformation parameters to
            test numerical convergence.
        ftol : float
            Tolerance on variations of the intensity comparison metric
            to test numerical convergence.
        gtol : float
            Tolerance on the gradient of the intensity comparison
            metric to test numerical convergence. Applicable to
            optimizers 'cg', 'ncg', 'bfgs' and 'steepest'.
        stepsize : float
            Step size to approximate the gradient and Hessian of the
            intensity comparison metric w.r.t. transformation
            parameters. Applicable to optimizers 'cg', 'ncg', 'bfgs'
            and 'steepest'.
        maxiter : int
            Maximum number of iterations in optimization.
        maxfun : int
            Maximum number of function evaluations in maxfun.
        """
        if between_loops is None:
            between_loops = loops
        t = realign4d(self._runs,
                      affine_class=self.affine_class,
                      time_interp=self._time_interp,
                      align_runs=align_runs,
                      loops=loops,
                      between_loops=between_loops,
                      speedup=speedup,
                      refscan=refscan,
                      borders=borders,
                      optimizer=optimizer,
                      xtol=xtol,
                      ftol=ftol,
                      gtol=gtol,
                      stepsize=stepsize,
                      maxiter=maxiter,
                      maxfun=maxfun)
        self._transforms, self._within_run_transforms,\
            self._mean_transforms = t

    def resample(self, r=None, align_runs=True):
        """
        Return the resampled run number r as a 4d nipy-like
        image. Returns all runs as a list of images if r is None.
        """
        if align_runs:
            transforms = self._transforms
        else:
            transforms = self._within_run_transforms
        runs = range(len(self._runs))
        if r is None:
            data = [resample4d(self._runs[r], transforms=transforms[r],
                               time_interp=self._time_interp) for r in runs]
            return [make_xyz_image(data[r], self._runs[r].affine, 'scanner')
                    for r in runs]
        else:
            data = resample4d(self._runs[r], transforms=transforms[r],
                              time_interp=self._time_interp)
            return make_xyz_image(data, self._runs[r].affine, 'scanner')


class SpaceTimeRealign(Realign4d):

    def __init__(self, images, tr, slice_times, slice_info,
                 affine_class=Rigid):
        """ Spatiotemporal realignment class for fMRI series.

        This class gives a high-level interface to :class:`Realign4d`

        Parameters
        ----------
        images : image or list of images
            Single or multiple input 4d images representing one or several fMRI
            runs.
        tr : None or float or "header-allow-1.0"
            Inter-scan repetition time in seconds, i.e. the time elapsed between
            two consecutive scans. If None, an attempt is made to read the TR
            from the header, but an exception is thrown for values 0 or 1. A
            value of "header-allow-1.0" will signal to accept a header TR of 1.
        slice_times : str or callable or array-like
            If str, one of the function names in ``SLICETIME_FUNCTIONS``
            dictionary from :mod:`nipy.algorithms.slicetiming.timefuncs`.  If
            callable, a function taking two parameters: ``n_slices`` and ``tr``
            (number of slices in the images, inter-scan repetition time in
            seconds). This function returns a vector of times of slice
            acquisition $t_i$ for each slice $i$ in the volumes.  See
            :mod:`nipy.algorithms.slicetiming.timefuncs` for a collection of
            functions for common slice acquisition schemes. If array-like, then
            should be a slice time vector as above.
        slice_info : int or length 2 sequence
            If int, the axis in `images` that is the slice axis.  In a 4D image,
            this will often be axis = 2.  If a 2 sequence, then elements are
            ``(slice_axis, slice_direction)``, where ``slice_axis`` is the slice
            axis in the image as above, and ``slice_direction`` is 1 if the
            slices were acquired slice 0 first, slice -1 last, or -1 if acquired
            slice -1 first, slice 0 last.  If `slice_info` is an int, assume
            ``slice_direction`` == 1.
        affine_class : ``Affine`` class, optional
            transformation class to use to calculate transformations between
            the volumes. Default is :class:``Rigid``
        """
        if tr is None:
            tr = tr_from_header(images)
            if tr == 1:
                raise ValueError('A TR of 1 was found in the header. '
                    'This value often stands in for an unknown TR. '
                    'Please specify TR explicitly. Alternatively '
                    'consider setting TR to "header-allow-1.0".')
        elif tr == "header-allow-1.0":
            tr = tr_from_header(images)
        if tr == 0:
            raise ValueError('Repetition time cannot be zero.')
        if slice_times is None:
            raise ValueError("slice_times must be set for space/time "
                             "registration; use SpaceRealign for space-only "
                             "registration")
        if slice_info is None:
            raise ValueError("slice_info cannot be None")
        try:
            len(slice_info)
        except TypeError:
            # Presumably an int
            slice_axis = slice_info
            slice_info = (slice_axis, 1)
        else: # sequence
            slice_axis, slice_direction = slice_info
        if type(images) in (list, tuple):
            n_slices = images[0].shape[slice_axis]
        else:
            n_slices = images.shape[slice_axis]
        if isinstance(slice_times, string_types):
            slice_times = timefuncs.SLICETIME_FUNCTIONS[slice_times]
        if hasattr(slice_times, '__call__'):
            slice_times = slice_times(n_slices, tr)
        self._init(images, tr, slice_times, slice_info, affine_class)


class SpaceRealign(Realign4d):

    def __init__(self, images, affine_class=Rigid):
        """ Spatial registration of time series with no time interpolation

        Parameters
        ----------
        images : image or list of images
            Single or multiple input 4d images representing one or several fMRI
            runs.
        affine_class : ``Affine`` class, optional
            transformation class to use to calculate transformations between
            the volumes. Default is :class:``Rigid``
        """
        self._init(images, 1., None, None, affine_class)


class FmriRealign4d(Realign4d):

    def __init__(self, images, slice_order=None,
                 tr=None, tr_slices=None, start=0.0,
                 interleaved=None, time_interp=None,
                 slice_times=None,
                 affine_class=Rigid, slice_info=None):
        """
        Spatiotemporal realignment class for fMRI series. This class
        is similar to `Realign4d` but provides a more flexible API for
        initialization in order to make it easier to declare slice
        acquisition times for standard sequences.

        Warning: this class is deprecated; please use :class:`SpaceTimeRealign`
        instead.

        Parameters
        ----------
        images : image or list of images
          Single or multiple input 4d images representing one or
          several fMRI runs.

        slice_order : str or array-like
          If str, one of {'ascending', 'descending'}. If array-like,
          then the order in which the slices were collected in
          time. For instance, the following represents an ascending
          contiguous sequence:

          slice_order = [0, 1, 2, ...]

          Note that `slice_order` differs from the argument used
          e.g. in the SPM slice timing routine in that it maps spatial
          slice positions to slice times. It is a mapping from space
          to time, while SPM conventionally uses the reverse mapping
          from time to space. For example, for an interleaved sequence
          with 10 slices, where we acquired slice 0 (in space) first,
          then slice 2 (in space) etc, `slice_order` would be [0, 5,
          1, 6, 2, 7, 3, 8, 4, 9]

          Using `slice_order` assumes that the inter-slice acquisition
          time is constant throughout acquisition. If this is not the
          case, use the `slice_times` argument instead and leave
          `slice_order` to None.

        tr : float
          Inter-scan repetition time, i.e. the time elapsed between
          two consecutive scans. The unit in which `tr` is given is
          arbitrary although it needs to be consistent with the
          `tr_slices` and `start` arguments if provided. If None, `tr`
          is computed internally assuming a regular slice acquisition
          scheme.

        tr_slices : float
          Inter-slice repetition time, same as `tr` for slices. If
          None, acquisition is assumed regular and `tr_slices` is set
          to `tr` divided by the number of slices.

        start : float
          Starting acquisition time (time of the first acquired slice)
          respective to the time origin for resampling. `start` is
          assumed to be given in the same unit as `tr`. Setting
          `start=0` means that the resampled data will be synchronous
          with the first acquired slice. Setting `start=-tr/2` means
          that the resampled data will be synchronous with the slice
          acquired at half repetition time.

        time_interp: bool
          Tells whether time interpolation is used or not within the
          realignment algorithm. If False, slices are considered to be
          acquired all at the same time, thus no slice timing
          correction will be performed.

        interleaved : bool
          Deprecated argument.

          Tells whether slice acquisition order is interleaved in a
          certain sense. Setting `interleaved` to True or False will
          trigger an error unless `slice_order` is 'ascending' or
          'descending' and `slice_times` is None.

          If slice_order=='ascending' and interleaved==True, the
          assumed slice order is (assuming 10 slices):

          [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]

          If slice_order=='descending' and interleaved==True, the
          assumed slice order is:

          [9, 4, 8, 3, 7, 2, 6, 1, 5, 0]

          WARNING: given that there exist other types of interleaved
          acquisitions depending on scanner settings and
          manufacturers, you should refrain from using the
          `interleaved` keyword argument unless you are sure what you
          are doing. It is generally safer to explicitly input
          `slice_order` or `slice_times`.

        slice_times : None, str or array-like

          This argument can be used instead of `slice_order`,
          `tr_slices`, `start` and `time_interp` altogether.

          If None, slices are assumed to be acquired simultaneously
          hence no slice timing correction is performed. If
          array-like, then `slice_times` gives the slice acquisition
          times along the slice axis in units that are consistent with
          the provided `tr`.

          Generally speaking, the following holds for sequences with
          constant inter-slice repetition time `tr_slices`:

          `slice_times` = `start` + `tr_slices` * `slice_order`

          For other sequences such as, e.g., sequences with
          simultaneously acquired slices, it is necessary to input
          `slice_times` explicitly along with `tr`.

        slice_info : None or tuple, optional
          None, or a tuple with slice axis as the first element and
          direction as the second, for instance (2, 1). If None, then
          the slice axis and direction are guessed from the first
          run's affine assuming that slices are collected along the
          closest axis to the z-axis. This means that we assume by
          default an axial acquisition with slice axis pointing from
          bottom to top of the head.
        """
        warnings.warn('Please use SpaceTimeRealign instead of this class; '
                      'We will soon remove this class',
                      FutureWarning,
                      stacklevel=2)
        # if slice_times not None, make sure that parameters redundant
        # with slice times all have their default value
        if slice_times is not None:
            if slice_order is not None \
                    or tr_slices is not None\
                    or start != 0.0 \
                    or time_interp is not None\
                    or interleaved is not None:
                raise ValueError('Attempting to set both `slice_times` '
                                 'and other arguments redundant with it')
            if tr is None:
                if len(slice_times) > 1:
                    tr = slice_times[-1] + slice_times[1] - 2 * slice_times[0]
                else:
                    tr = 2 * slice_times[0]
                warnings.warn('No `tr` entered. Assuming regular acquisition'
                              ' with tr=%f' % tr)
        # case where slice_time is None
        else:
            # assume regular slice acquisition, therefore tr is
            # arbitrary
            if tr is None:
                tr = 1.0
            # if no slice order provided, assume synchronous slices
            if slice_order is None:
                if not time_interp == False:
                    raise ValueError('Slice order is requested '
                                     'with time interpolation switched on')
                slice_times = 0.0
            else:
                # if slice_order is a key word, replace it with the
                # appropriate array of slice indices
                if slice_order in ('ascending', 'descending'):
                    if isinstance(images, (list, tuple, np.array)):
                        xyz_img = as_xyz_image(images[0])
                    else:
                        xyz_img = as_xyz_image(images)

                    slice_axis, _ = guess_slice_axis_and_direction(
                        slice_info, xyz_affine(xyz_img))
                    nslices = xyz_img.shape[slice_axis]
                    if interleaved:
                        warnings.warn('`interleaved` keyword argument is '
                                      'deprecated',
                                      FutureWarning,
                                      stacklevel=2)
                        aux = np.argsort(list(range(0, nslices, 2)) +
                                         list(range(1, nslices, 2)))
                    else:
                        aux = np.arange(nslices)
                    if slice_order == 'descending':
                        aux = aux[::-1]
                    slice_order = aux
                # if slice_order is provided explicitly, issue a
                # warning and make sure interleaved is set to None
                else:
                    warnings.warn('Please make sure you are NOT using '
                                  'SPM-style slice order declaration')
                    if interleaved is not None:
                        raise ValueError('`interleaved` should be None when '
                                         'providing explicit slice order')
                    slice_order = np.asarray(slice_order)
                if tr_slices is None:
                    tr_slices = float(tr) / float(len(slice_order))
                if start is None:
                    start = 0.0
                slice_times = start + tr_slices * slice_order

        self._init(images, tr, slice_times, slice_info, affine_class)
