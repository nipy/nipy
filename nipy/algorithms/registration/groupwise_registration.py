# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from ..utils.affines import apply_affine
import numpy as np
from scipy.optimize import (fmin as fmin_simplex,
                            fmin_powell,
                            fmin_cg,
                            fmin_bfgs,
                            fmin_ncg)

from ...core.image.affine_image import AffineImage
from ..optimize import fmin_steepest
from .affine import Rigid
from ._registration import (_cspline_transform,
                            _cspline_sample3d,
                            _cspline_sample4d)


# Module globals
VERBOSE = True  # enables online print statements
SLICE_ORDER = 'ascending'
INTERLEAVED = False
SLICE_AXIS = 2
OPTIMIZER = 'ncg'
XTOL = 1e-5
FTOL = 1e-5
GTOL = 1e-5
STEPSIZE = 1e-6
MAXITER = 64
MAXFUN = None
LOOPS = 5, 1  # loops within each run
BETWEEN_LOOPS = 5, 1  # loops used to realign different runs
SPEEDUP = 5, 2
BORDERS = 1, 1, 1
REFSCAN = 0
EXTRAPOLATE_SPACE = 'reflect'
EXTRAPOLATE_TIME = 'reflect'
TINY = float(np.finfo(np.double).tiny)


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
    slices = [slice(b, d - b, s) for d, s, b in zip(dims, subsampling, borders)]
    xyz = np.mgrid[slices]
    xyz = np.rollaxis(xyz, 0, 4)
    xyz = np.reshape(xyz, [np.prod(xyz.shape[0:-1]), 3])
    return xyz


class Image4d(object):
    """
    Class to represent a sequence of 3d scans (possibly acquired on a
    slice-by-slice basis).
    """
    def __init__(self, array, affine, tr, tr_slices=None, start=0.0,
                 slice_order=SLICE_ORDER, interleaved=INTERLEAVED,
                 slice_axis=SLICE_AXIS):
        """
        Configure fMRI acquisition time parameters.

        tr  : inter-scan repetition time, i.e. the time elapsed
              between two consecutive scans
        tr_slices : inter-slice repetition time, same as tr for slices
        start   : starting acquisition time respective to the implicit
                  time origin
        slice_order : string or array
        """
        self.array = array
        self.affine = affine
        nslices = array.shape[slice_axis]

        # Default slice repetition time (no silence)
        if tr_slices == None:
            tr_slices = tr / float(nslices)

        # Set slice order
        if isinstance(slice_order, str):
            if not interleaved:
                aux = range(nslices)
            else:
                p = nslices / 2
                aux = []
                for i in range(p):
                    aux.extend([i, p+i])
                if nslices % 2:
                    aux.append(nslices - 1)
            if slice_order == 'descending':
                aux.reverse()
            slice_order = aux

        # Set timing values
        self.nslices = nslices
        self.tr = float(tr)
        self.tr_slices = float(tr_slices)
        self.start = float(start)
        self.slice_order = np.asarray(slice_order)
        self.interleaved = bool(interleaved)
        # assume that the world referential is 'scanner' as defined by
        # the nifti norm
        self.reversed_slices = affine[slice_axis][slice_axis] < 0

    def z_to_slice(self, z):
        """
        Account for the fact that slices may be stored in reverse
        order wrt the scanner coordinate system convention (slice 0 ==
        bottom of the head)
        """
        if self.reversed_slices:
            return self.nslices - 1 - z
        else:
            return z

    def scanner_time(self, zv, t):
        """
        tv = scanner_time(zv, t)
        zv, tv are grid coordinates; t is an actual time value.
        """
        corr = self.tr_slices * interp_slice_order(self.z_to_slice(zv), self.slice_order)
        return (t - self.start-corr) / self.tr


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
        self.dims = im4d.array.shape
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
            self.cbspline = _cspline_transform(im4d.array)
        else:
            self.cbspline = np.zeros(self.dims, dtype='double')
            for t in range(self.dims[3]):
                self.cbspline[:, :, :, t] = _cspline_transform(im4d.array[:, :, :, t])
        # The reference scan conventionally defines the head
        # coordinate system
        self.refscan = refscan
        # Set the minimization method
        self.set_fmin(optimizer, stepsize, xtol, ftol, gtol, maxiter, maxfun)
        self.optimize_template = optimize_template
        # Auxiliary array for realignment estimation
        self._res = np.zeros(masksize, dtype='double')
        self._res0 = np.zeros(masksize, dtype='double')
        self._aux = np.zeros(masksize, dtype='double')
        self.A = np.zeros((masksize, self.transforms[0].param.size), dtype='double')
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
                _cspline_sample4d(res[:, :, :, t], self.cbspline, X, Y, Z, T, mt='nearest')
            else:
                _cspline_sample3d(res[:, :, :, t], self.cbspline[:, :, :, t], X, Y, Z)
        return res

    def set_fmin(self, optimizer, stepsize, xtol, ftol, gtol, maxiter, maxfun):
        """
        Return the minimization function.
        """
        self.stepsize = stepsize
        self.fmin_use_derivatives = True
        self.fmin_args = []
        if optimizer == 'simplex':
            self.fmin_kwargs = {'xtol': xtol,
                                'ftol': ftol,
                                'maxiter': maxiter,
                                'maxfun': maxfun}
            self.fmin = fmin_simplex
            self.fmin_use_derivatives = False
        elif optimizer == 'powell':
            self.fmin_kwargs = {'xtol': xtol,
                                'ftol': ftol,
                                'maxiter': maxiter,
                                'maxfun': maxfun}
            self.fmin = fmin_powell
            self.fmin_use_derivatives = False
        elif optimizer == 'cg':
            self.fmin_kwargs = {'gtol': gtol,
                                'maxiter': maxiter,
                                'fprime': None}
            self.fmin = fmin_cg
        elif optimizer == 'bfgs':
            self.fmin_kwargs = {'gtol': gtol,
                                'maxiter': maxiter,
                                'fprime': None}
            self.fmin = fmin_bfgs
        elif optimizer == 'ncg':
            self.fmin_args = ['fprime']
            self.fmin_kwargs = {'avextol': xtol,
                                'maxiter': maxiter,
                                'fhess': None}
            self.fmin = fmin_ncg
        elif optimizer == 'steepest':
            self.fmin_kwargs = {'xtol': xtol,
                                'ftol': ftol,
                                'maxiter': maxiter,
                                'fprime': None}
            self.fmin = fmin_steepest
        else:
            raise ValueError('unknown optimizer: %s' % optimizer)

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
        self._V = np.maximum(self.offset + np.mean(self._res ** 2), TINY)
        self._res0[:] = self.data[:, self._t] - self.mu0
        self._V0 = np.maximum(self.offset0 + np.mean(self._res0 ** 2), TINY)
        if self.fmin_use_derivatives:
            # linearize the data wrt the transform parameters
            # use the auxiliary array to save the current resampled data
            self._aux[:] = self.data[:, self._t]
            basis = np.eye(6)
            for j in range(pc.size):
                self.set_transform(self._t, pc + self.stepsize * basis[j])
                self.A[:, j] = (self.data[:, self._t] - self._aux) / self.stepsize
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
        The alignment energy is defined as half the ratio between the
        average temporal variance in the sequence and the global
        spatio-temporal variance.
        """
        #  return np.mean(self._res**2)
        return np.log(self._V / self._V0)

    def _energy_gradient(self):
        #return (2/float(self.data.shape[0]))*np.dot(self.A.T, self._res)
        return self._dV / self._V - self._dV0 / self._V0

    def _energy_hessian(self):
        # return (2/float(self.data.shape[0]))*np.dot(self.A.T, self.A)
        return (1 / self._V - 1 / self._V0) * self._H - np.dot(self._dV, self._dV.T) / (self._V ** 2) + np.dot(self._dV0, self._dV0.T) / (self._V0 ** 2)

    def estimate_instant_motion(self, t):
        """
        Estimate motion parameters at a particular time.
        """
        if VERBOSE:
            print('Estimating motion at time frame %d/%d...' % (t + 1, self.nscans))

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

        args = []
        if 'fprime' in self.fmin_args:
            args += [fprime]
        if 'fprime' in self.fmin_kwargs:
            self.fmin_kwargs['fprime'] = fprime
        if 'fhess' in self.fmin_kwargs:
            self.fmin_kwargs['fhess'] = fhess

        pc = self.fmin(f, self.transforms[t].param, *args, **self.fmin_kwargs)
        self.set_transform(t, pc)

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
            self.mu = self.data[:,self.refscan].copy()
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
        Tref_inv = self.transforms[self.refscan].inv()
        for t in range(self.nscans):
            self.transforms[t] = (self.transforms[t]).compose(Tref_inv)


def resample4d(im4d, transforms, time_interp=True):
    """
    Resample a 4D image according to the specified sequence of spatial
    transforms, using either 4D interpolation if `time_interp` is True
    and 3D interpolation otherwise.
    """
    r = Realign4dAlgorithm(im4d, transforms=transforms, time_interp=time_interp)
    return r.resample_full_data()


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
                         maxfun=MAXFUN):
    """
    Realign a single run in space and time.

    Parameters
    ----------
    im4d : Image4d instance

    speedup : int or sequence
      If a sequence, implement a multi-scale

    """
    if not hasattr(loops, '__iter__'):
        loops = [loops]
    repeats = len(loops)

    def format_arg(x):
        if not hasattr(x, '__iter__'):
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
    for loops_, speedup_, optimizer_, xtol_, ftol_, gtol_, stepsize_, maxiter_, maxfun_ in opt_params:
        subsampling = adjust_subsampling(speedup_, im4d.array.shape[0:3])
        r = Realign4dAlgorithm(im4d,
                               transforms=transforms,
                               affine_class=affine_class,
                               time_interp=time_interp,
                               subsampling=subsampling,
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
    if not hasattr(runs, '__iter__'):
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
                                       maxfun=maxfun) for run in runs]
    if not align_runs:
        return transforms, transforms, None

    # Correct between-session motion using the mean image of each
    # corrected run, and creating a fake time series with no temporal
    # smoothness
    ## FIXME: check that all runs have the same to-world transform
    corr_runs = [resample4d(runs[i], transforms=transforms[i], time_interp=time_interp) for i in range(nruns)]
    aux = np.rollaxis(np.asarray([c.mean(3) for c in corr_runs]), 0, 4)
    mean_img = Image4d(aux, affine=runs[0].affine, tr=1.0, tr_slices=0.0)
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


def split_affine(a):
    # that's a horrible hack until we fix the inconsistency between
    # Image and AffineImage
    sa = np.eye(4)
    sa[0:3, 0:3] = a[0:3, 0:3]
    if a.shape[1] > 4:
        sa[0:3, 3] = a[0:3, 4]
    return sa, a[3, 3]


class Realign4d(object):

    def __init__(self, images, affine_class=Rigid):
        self._generic_init(images, affine_class, SLICE_ORDER, INTERLEAVED,
                           1.0, 0.0, 0.0, False)

    def _generic_init(self, images, affine_class,
                      slice_order, interleaved, tr, tr_slices,
                      start, time_interp):
        if not hasattr(images, '__iter__'):
            images = [images]
        self._runs = []
        self.affine_class = affine_class
        for im in images:
            affine, _tr = split_affine(im.affine)
            if tr == None:
                tr = _tr
            self._runs.append(Image4d(im.get_data(), affine, tr=tr, tr_slices=tr_slices,
                                      start=start, slice_order=slice_order, interleaved=interleaved))
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
                 maxfun=MAXFUN):
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
                      maxfun=maxfun)
        self._transforms, self._within_run_transforms, self._mean_transforms = t

    def resample(self, align_runs=True):
        """
        Return a list of 4d nipy-like images corresponding to the
        resampled runs.
        """
        if align_runs:
            transforms = self._transforms
        else:
            transforms = self._within_run_transforms
        runs = range(len(self._runs))
        data = [resample4d(self._runs[r], transforms=transforms[r], time_interp=self._time_interp) for r in runs]
        return [AffineImage(data[r], self._runs[r].affine, 'scanner') for r in runs]


class FmriRealign4d(Realign4d):

    def __init__(self, images, slice_order, interleaved,
                 tr=None, tr_slices=None,
                 start=0.0, time_interp=True,  affine_class=Rigid):
        self._generic_init(images, affine_class, slice_order, interleaved,
                           tr, tr_slices,
                           start, time_interp)
