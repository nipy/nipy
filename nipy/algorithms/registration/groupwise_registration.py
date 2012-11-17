# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import warnings
import numpy as np

from nibabel.affines import apply_affine

from ...fixes.nibabel import io_orientation

from ...core.image.image_spaces import (make_xyz_image,
                                        xyz_affine,
                                        as_xyz_image)
from .optimizer import configure_optimizer, use_derivatives
from .affine import Rigid
from ._registration import (_cspline_transform,
                            _cspline_sample3d,
                            _cspline_sample4d)


# Module globals
VERBOSE = True  # enables online print statements
SLICE_ORDER = 'ascending'
INTERLEAVED = None
OPTIMIZER = 'ncg'
XTOL = 1e-5
FTOL = 1e-5
GTOL = 1e-5
STEPSIZE = 1e-6
SMALL = 1e-20
MAXITER = 64
MAXFUN = None
BORDERS = 1, 1, 1
REFSCAN = 0
EXTRAPOLATE_SPACE = 'reflect'
EXTRAPOLATE_TIME = 'reflect'

LOOPS = 5  # loops within each run
BETWEEN_LOOPS = 5  # loops used to realign different runs
SPEEDUP = 5  # image sub-sampling factor for speeding up
"""
# How to tune those parameters for a multi-resolution implementation
LOOPS = 5, 1
BETWEEN_LOOPS = 5, 1
SPEEDUP = 5, 2
"""


def interp_slice_order(Z, slice_order):
    Z = np.asarray(Z)
    nslices = len(slice_order)
    aux = np.asarray(list(slice_order) + [slice_order[0] + nslices])
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


class Image4d(object):
    """
    Class to represent a sequence of 3d scans (possibly acquired on a
    slice-by-slice basis).

    Object remains empty until the data array is actually loaded in memory.

    Parameters
    ----------
      data : nd array or proxy (function that actually gets the array)
    """
    def __init__(self, data, affine, tr, tr_slices=None, start=0.0,
                 slice_order=SLICE_ORDER, interleaved=INTERLEAVED,
                 slice_info=None):
        """
        Configure fMRI acquisition time parameters.
        """
        self.affine = np.asarray(affine)
        self.tr = float(tr)
        self.start = float(start)
        self.interleaved = bool(interleaved)

        # guess the slice axis and direction (z-axis)
        if slice_info == None:
            orient = io_orientation(self.affine)
            self.slice_axis = int(np.where(orient[:, 0] == 2)[0])
            self.slice_direction = int(orient[self.slice_axis, 1])
        else:
            self.slice_axis = int(slice_info[0])
            self.slice_direction = int(slice_info[1])

        # unformatted parameters
        self._tr_slices = tr_slices
        self._slice_order = slice_order

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
        if self._data == None:
            self._load_data()
        return self._data
    
    def get_shape(self):
        if self._shape == None:
            self._load_data()
        return self._shape

    def _init_timing_parameters(self):
        # Number of slices
        nslices = self.get_shape()[self.slice_axis]
        self.nslices = nslices
        # Default slice repetition time (no silence)
        if self._tr_slices == None:
            self.tr_slices = self.tr / float(nslices)
        else:
            self.tr_slices = float(self._tr_slices)
        # Set slice order
        if isinstance(self._slice_order, str):
            if not self.interleaved:
                aux = range(nslices)
            else:
                aux = range(nslices)[0::2] + range(nslices)[1::2]
            if self._slice_order == 'descending':
                aux.reverse()
            self.slice_order = np.array(aux)
        else:
            # Verify correctness of provided slice indexes
            provided_slices = np.array(sorted(self._slice_order))
            if np.any(provided_slices != np.arange(nslices)):
                raise ValueError(
                    "Incorrect slice indexes were provided. There are %d "
                    "slices in the volume, indexes should start from 0 and "
                    "list all slices. "
                    "Provided slice_order: %s" % (nslices, self._slice_order))
            self.slice_order = np.asarray(self._slice_order)

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
        corr = self.tr_slices * interp_slice_order(self.z_to_slice(zv),
                                                   self.slice_order)
        return (t - self.start - corr) / self.tr

    def free_data(self):
        if not self._get_data == None:
            self._data = None


class Realign4dAlgorithm(object):

    def __init__(self,
                 im4d,
                 affine_class=Rigid,
                 transforms=None,
                 time_interp=True,
                 subsampling=(1, 1, 1),
                 borders=BORDERS,
                 optimizer=OPTIMIZER,
                 optimize_template=True,
                 xtol=XTOL,
                 ftol=FTOL,
                 gtol=GTOL,
                 stepsize=STEPSIZE,
                 maxiter=MAXITER,
                 maxfun=MAXFUN,
                 refscan=REFSCAN):

        self.dims = im4d.get_shape()
        self.nscans = self.dims[3]
        self.xyz = make_grid(self.dims[0:3], subsampling, borders)
        masksize = self.xyz.shape[0]
        self.data = np.zeros([masksize, self.nscans], dtype='double')

        # Initialize space/time transformation parameters
        self.affine = im4d.affine
        self.inv_affine = np.linalg.inv(self.affine)
        if transforms == None:
            self.transforms = [affine_class() for scan in range(self.nscans)]
        else:
            self.transforms = transforms
        self.scanner_time = im4d.scanner_time
        self.timestamps = im4d.tr * np.arange(self.nscans)

        # Compute the 4d cubic spline transform
        self.time_interp = time_interp
        if time_interp:
            self.cbspline = _cspline_transform(im4d.get_data())
        else:
            self.cbspline = np.zeros(self.dims, dtype='double')
            for t in range(self.dims[3]):
                self.cbspline[:, :, :, t] =\
                    _cspline_transform(im4d.get_data()[:, :, :, t])

        # The reference scan conventionally defines the head
        # coordinate system
        self.optimize_template = optimize_template
        if not optimize_template and refscan == None:
            self.refscan = REFSCAN
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
                              mx=EXTRAPOLATE_SPACE,
                              my=EXTRAPOLATE_SPACE,
                              mz=EXTRAPOLATE_SPACE,
                              mt=EXTRAPOLATE_TIME)
        else:
            _cspline_sample3d(self.data[:, t],
                              self.cbspline[:, :, :, t],
                              X, Y, Z,
                              mx=EXTRAPOLATE_SPACE,
                              my=EXTRAPOLATE_SPACE,
                              mz=EXTRAPOLATE_SPACE)

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
        fixed = range(self.nscans)
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
            pc = fmin(f, self.transforms[t].param, disp=VERBOSE, *args, **kwargs)
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
        if self.refscan == None:
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
                         loops=LOOPS,
                         speedup=SPEEDUP,
                         borders=BORDERS,
                         optimizer=OPTIMIZER,
                         xtol=XTOL,
                         ftol=FTOL,
                         gtol=GTOL,
                         stepsize=STEPSIZE,
                         maxiter=MAXITER,
                         maxfun=MAXFUN,
                         refscan=REFSCAN):
    """
    Realign a single run in space and time.

    Parameters
    ----------
    im4d : Image4d instance

    speedup : int or sequence
      If a sequence, implement a multi-scale

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
                               borders=borders,
                               refscan=refscan,
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
              loops=LOOPS,
              between_loops=BETWEEN_LOOPS,
              speedup=SPEEDUP,
              borders=BORDERS,
              optimizer=OPTIMIZER,
              xtol=XTOL,
              ftol=FTOL,
              gtol=GTOL,
              stepsize=STEPSIZE,
              maxiter=MAXITER,
              maxfun=MAXFUN,
              refscan=REFSCAN):
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
                                       borders=borders,
                                       optimizer=optimizer,
                                       xtol=xtol,
                                       ftol=ftol,
                                       gtol=gtol,
                                       stepsize=stepsize,
                                       maxiter=maxiter,
                                       maxfun=maxfun,
                                       refscan=refscan) for run in runs]
    if not align_runs:
        return transforms, transforms, None

    # Correct between-session motion using the mean image of each
    # corrected run, and creating a fake time series with no temporal
    # smoothness
    ## FIXME: check that all runs have the same to-world transform
    mean_img_shape = list(runs[0].get_shape()[0:3]) + [nruns]
    mean_img_data = np.zeros(mean_img_shape)

    for i in range(nruns):
        corr_run = resample4d(runs[i], transforms=transforms[i],
                              time_interp=time_interp)
        mean_img_data[..., i] = corr_run.mean(3)
    del corr_run

    mean_img = Image4d(mean_img_data, affine=runs[0].affine,
                       tr=1.0, tr_slices=0.0)
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

    def __init__(self, images, affine_class=Rigid):
        self._generic_init(images, affine_class, SLICE_ORDER, INTERLEAVED,
                           1.0, 0.0, 0.0, False, None)

    def _generic_init(self, images, affine_class,
                      slice_order, interleaved, tr, tr_slices,
                      start, time_interp, slice_info):
        if slice_order == None:
            slice_order = SLICE_ORDER
            if time_interp:
                raise ValueError('Slice order is requested'
                          + ' with time interpolation switched on')
            time_interp = False
        if tr == None:
            raise ValueError('Repetition time cannot be None')
        if not type(images) in (list, tuple, np.array):
            images = [images]
        self._runs = []
        self.affine_class = affine_class
        for im in images:
            xyz_img = as_xyz_image(im)
            self._runs.append(Image4d(xyz_img.get_data,
                                      xyz_affine(xyz_img),
                                      tr=tr, tr_slices=tr_slices,
                                      start=start, slice_order=slice_order,
                                      interleaved=interleaved,
                                      slice_info=slice_info))
        self._transforms = [None for run in self._runs]
        self._within_run_transforms = [None for run in self._runs]
        self._mean_transforms = [None for run in self._runs]
        self._time_interp = time_interp

    def estimate(self,
                 loops=LOOPS,
                 between_loops=None,
                 align_runs=True,
                 speedup=SPEEDUP,
                 borders=BORDERS,
                 optimizer=OPTIMIZER,
                 xtol=XTOL,
                 ftol=FTOL,
                 gtol=GTOL,
                 stepsize=STEPSIZE,
                 maxiter=MAXITER,
                 maxfun=MAXFUN,
                 refscan=REFSCAN):
        if between_loops == None:
            between_loops = loops
        t = realign4d(self._runs,
                      affine_class=self.affine_class,
                      time_interp=self._time_interp,
                      align_runs=align_runs,
                      loops=loops,
                      between_loops=between_loops,
                      speedup=speedup,
                      borders=borders,
                      optimizer=optimizer,
                      xtol=xtol,
                      ftol=ftol,
                      gtol=gtol,
                      stepsize=stepsize,
                      maxiter=maxiter,
                      maxfun=maxfun,
                      refscan=refscan)
        self._transforms, self._within_run_transforms,\
            self._mean_transforms = t

    def resample(self, r=None, align_runs=True):
        """
        Return the resampled run number r as a 4d nipy-like
        image. Returns all runs as a list of images if r == None.
        """
        if align_runs:
            transforms = self._transforms
        else:
            transforms = self._within_run_transforms
        runs = range(len(self._runs))
        if r == None:
            data = [resample4d(self._runs[r], transforms=transforms[r],
                               time_interp=self._time_interp) for r in runs]
            return [make_xyz_image(data[r], self._runs[r].affine, 'scanner')
                    for r in runs]
        else:
            data = resample4d(self._runs[r], transforms=transforms[r],
                              time_interp=self._time_interp)
            return make_xyz_image(data, self._runs[r].affine, 'scanner')


class FmriRealign4d(Realign4d):

    def __init__(self, images, slice_order, interleaved=None,
                 tr=1.0, tr_slices=None, start=0.0, time_interp=True,
                 affine_class=Rigid, slice_info=None):

        """
        Spatiotemporal realignment class for fMRI series.

        Parameters
        ----------
        images : image or list of images
          Single or multiple input 4d images representing one or
          several fMRI runs.

        tr : float
          Inter-scan repetition time, i.e. the time elapsed between
          two consecutive scans. The unit in which `tr` is given is
          arbitrary although it needs to be consistent with the
          `tr_slices` and `start` arguments.

        tr_slices : float
          Inter-slice repetition time, same as tr for slices. If None,
          acquisition is assumed continuous and `tr_slices` is set to
          `tr` divided by the number of slices.

        start : float
          Starting acquisition time respective to the implicit time
          origin.

        slice_order : str or array-like
          If str, one of {'ascending', 'descending'}. If array-like,
          then the order in which the slices were collected in
          time. For instance, the following represents an ascending
          contiguous sequence:

          slice_order = [0, 1, 2, ...]

        interleaved : bool
          Deprecated.

          Whether slice acquisition order is interleaved. Ignored if
          `slice_order` is array-like.

          If slice_order=='ascending' and interleaved==True, the
          assumed slice order is:

          [0, 2, 4, ..., 1, 3, 5, ...]

          If slice_order=='descending' and interleaved==True, the
          assumed slice order is:

          [N-1, N-3, N-5, ..., N-2, N-4, N-6]

          Given that there exist other types of interleaved
          acquisitions depending on scanner settings and
          manufacturers, it is strongly recommended to input the
          slice_order as an array unless you are sure what you are
          doing.

        slice_info : None or tuple, optional
          None, or a tuple with slice axis as the first element and
          direction as the second, for instance (2, 1).  If None, then
          guess the slice axis, and direction, as the closest to the z
          axis, as estimated from the affine.
        """
        if not interleaved == None:
            warnings.warn('interleaved keyword is deprecated. Please input explicit slice order instead.')
        self._generic_init(images, affine_class, slice_order, interleaved,
                           tr, tr_slices, start, time_interp, slice_info)
