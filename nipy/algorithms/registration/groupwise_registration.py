# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from .affine import Rigid, Similarity, Affine, apply_affine
from ._cubic_spline import cspline_transform, cspline_sample3d, cspline_sample4d

from nipy.core.image.affine_image import AffineImage
from nipy.algorithms.optimize import fmin_steepest

import numpy as np
from scipy.optimize import fmin as fmin_simplex, fmin_powell, fmin_cg, fmin_bfgs

# Module globals 
VERBOSE = True # enables online print statements
OPTIMIZER = 'powell'
XTOL = 1e-2
FTOL = 1e-2
GTOL = 1e-2
STEPSIZE = 1e-1
MAXITER = 5
SLICE_ORDER = 'ascending'
INTERLEAVED = False
SLICE_AXIS = 2 
SPEEDUP = 4
LOOPS = 2 # loops within each run 
BETWEEN_LOOPS = 5 
METRIC = 'sad'
REFSCAN = 0 

def interp_slice_order(Z, slice_order): 
    Z = np.asarray(Z)
    nslices = len(slice_order)
    aux = np.asarray(list(slice_order)+[slice_order[0]+nslices])
    Zf = np.floor(Z).astype('int')
    w = Z - Zf
    Zal = Zf % nslices
    Za = Zal + w
    ret = (1-w)*aux[Zal] + w*aux[Zal+1]
    ret += (Z-Za)
    return ret

def scanner_coords(xyz, affine, from_world, to_world):
    Tv = np.dot(from_world, np.dot(affine, to_world))
    XYZ = apply_affine(Tv, xyz)
    return XYZ[:,0], XYZ[:,1], XYZ[:,2]

def make_grid(dims, speedup=1):
    s = max(1, int(speedup))
    slices = [slice(s-1, d+1-s, s) for d in dims]
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
            tr_slices = tr/float(nslices)

        # Set slice order
        if isinstance(slice_order, str): 
            if not interleaved:
                aux = range(nslices)
            else:
                p = nslices/2
                aux = []
                for i in range(p):
                    aux.extend([i,p+i])
                if nslices%2:
                    aux.append(nslices-1)
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
        self.reversed_slices = affine[slice_axis][slice_axis]<0 

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
        corr = self.tr_slices*interp_slice_order(self.z_to_slice(zv), self.slice_order)
        return (t-self.start-corr)/self.tr



class Realign4dAlgorithm(object):

    def __init__(self, 
                 im4d, 
                 speedup=SPEEDUP,
                 optimizer=OPTIMIZER, 
                 affine_class=Rigid,
                 transforms=None, 
                 time_interp=True, 
                 metric=METRIC, 
                 refscan=REFSCAN):
        self.dims = im4d.array.shape
        self.nscans = self.dims[3]
        self.xyz = make_grid(self.dims[0:3], speedup)
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
        self.timestamps = im4d.tr*np.arange(self.nscans)
        # Compute the 4d cubic spline transform
        self.time_interp = time_interp 
        if time_interp: 
            self.cbspline = cspline_transform(im4d.array)
        else: 
            self.cbspline = np.zeros(self.dims)
            for t in range(self.dims[3]): 
                self.cbspline[:,:,:,t] = cspline_transform(im4d.array[:,:,:,t])
        # Intensity comparison metric 
        self.diffs = np.zeros(masksize)
        if metric == 'ssd':
            self.make_template = lambda: np.mean(self.data, 1)
            self.metric = lambda d: np.mean(d**2) 
        elif metric == 'sad':
            self.make_template = lambda: np.median(self.data, 1)
            self.metric = lambda d: np.mean(np.abs(d)) 
        else:
            raise ValueError('unknown metric')
        # The reference scan conventionally defines the head
        # coordinate system
        self.refscan = refscan 
        # Set the optimization (minimization) method
        self.set_fmin(optimizer)

    def compute_metric(self, t):
        """
        Intensity difference metric
        """
        self.resample(t)
        self.diffs[:] = self.data[:,t] - self.template
        return self.metric(self.diffs)

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
            cspline_sample4d(self.data[:,t], self.cbspline, X, Y, Z, T, mt=1)
        else: 
            cspline_sample3d(self.data[:,t], self.cbspline[:,:,:,t], X, Y, Z)

    def set_fmin(self, optimizer): 
        """
        Return the minimization function. 
        """
        if optimizer=='powell':
            self.fmin_kwargs = {'xtol':XTOL, 'ftol':FTOL}
            self.fmin = fmin_powell
        elif optimizer=='steepest':
            self.fmin_kwargs = {'xtol':XTOL, 'ftol': FTOL, 'step':STEPSIZE}
            self.fmin = fmin_steepest
        elif optimizer=='cg':
            self.fmin_kwargs = {'gtol':GTOL, 'maxiter':MAXITER}
            self.fmin = fmin_cg
        elif optimizer=='bfgs':
            self.fmin_kwargs = {'gtol':GTOL, 'maxiter':MAXITER}
            self.fmin = fmin_bfgs
        elif optimizer=='simplex':
            self.fmin_kwargs = {'xtol':XTOL, 'ftol':FTOL}
            self.fmin = fmin_simplex
        else: 
            raise ValueError('unknown optimizer')


    def estimate_motion_at_time(self, t, update_template=True):
        """ 
        Estimate motion at a particular time frame.
        """
        if VERBOSE: 
            print('Estimating motion at time frame %d/%d...' % (t+1, self.nscans))
        def cost(pc): 
            self.transforms[t].param = pc
            return self.compute_metric(t)
        if update_template: 
            self.template = self.make_template()
        self.transforms[t].param = self.fmin(cost, self.transforms[t].param, **self.fmin_kwargs)
        self.resample(t)


    def estimate_motion(self, update_template=True):
        """
        Estimate motion parameters for the whole sequence. 
        """

        # Resample all time frames according to the current space/time
        # transformation
        for t in range(self.nscans):
            if VERBOSE:
                print('Resampling scan %d/%d' % (t+1, self.nscans))
            self.resample(t)

        # Optimize motion parameters 
        for t in range(self.nscans):
            self.estimate_motion_at_time(t, update_template=update_template)
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


    def resample_full_data(self):
        if VERBOSE:
            print('Gridding...')
        xyz = make_grid(self.dims[0:3])
        res = np.zeros(self.dims)
        for t in range(self.nscans):
            if VERBOSE:
                print('Fully resampling scan %d/%d' % (t+1, self.nscans))
            X, Y, Z = scanner_coords(xyz, self.transforms[t].as_affine(), 
                                     self.inv_affine, self.affine)
            if self.time_interp: 
                T = self.scanner_time(Z, self.timestamps[t])
                cspline_sample4d(res[:,:,:,t], self.cbspline, X, Y, Z, T, mt=1)
            else: 
                cspline_sample3d(res[:,:,:,t], self.cbspline[:,:,:,t], X, Y, Z)
        return res
    


def resample4d(im4d, transforms, time_interp=True): 
    """
    corr_im4d_array = resample4d(im4d, transforms=None, time_interp=True)
    """
    r = Realign4dAlgorithm(im4d, transforms=transforms, time_interp=time_interp)
    return r.resample_full_data()



def single_run_realign4d(im4d, 
                         loops=LOOPS, 
                         speedup=SPEEDUP, 
                         optimizer=OPTIMIZER, 
                         affine_class=Rigid, 
                         time_interp=True, 
                         metric=METRIC): 
    """
    transforms = single_run_realign4d(im4d, loops=2, speedup=4, optimizer='powell', time_interp=True)

    Parameters
    ----------
    im4d : Image4d instance

    """ 
    r = Realign4dAlgorithm(im4d, 
                           speedup=speedup, 
                           optimizer=optimizer, 
                           time_interp=time_interp, 
                           affine_class=affine_class, 
                           metric=metric)
    for loop in range(loops): 
        r.estimate_motion()
        r.align_to_refscan()
    return r.transforms

def realign4d(runs, 
              loops=LOOPS, 
              between_loops=BETWEEN_LOOPS, 
              speedup=SPEEDUP, 
              optimizer=OPTIMIZER, 
              align_runs=True, 
              time_interp=True, 
              affine_class=Rigid,
              metric=METRIC): 
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
    transforms = [single_run_realign4d(run, loops=loops, 
                                       speedup=speedup, optimizer=optimizer,
                                       time_interp=time_interp, 
                                       affine_class=affine_class,
                                       metric=metric) for run in runs]
    if not align_runs: 
        return transforms, transforms, None

    # Correct between-session motion using the mean image of each corrected run 
    corr_runs = [resample4d(runs[i], transforms=transforms[i], time_interp=time_interp) for i in range(nruns)]
    aux = np.rollaxis(np.asarray([c.mean(3) for c in corr_runs]), 0, 4)
    ## Fake time series with zero inter-slice time 
    ## FIXME: check that all runs have the same to-world transform
    mean_img = Image4d(aux, affine=runs[0].affine, tr=1.0, tr_slices=0.0) 
    transfo_mean = single_run_realign4d(mean_img, loops=between_loops, speedup=speedup, 
                                        optimizer=optimizer, time_interp=time_interp, 
                                        metric=metric)

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
    return sa, a[3,3]


class Realign4d(object): 

    def __init__(self, images, affine_class=Rigid, metric=METRIC):
        self._generic_init(images, affine_class, SLICE_ORDER, INTERLEAVED, 
                           1.0, 0.0, 0.0, False, metric)

    def _generic_init(self, images, affine_class, 
                      slice_order, interleaved, tr, tr_slices, 
                      start, time_interp, metric):
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
        self.metric = metric

    def estimate(self, loops=LOOPS, between_loops=None, align_runs=True, 
                 speedup=SPEEDUP, optimizer=OPTIMIZER): 
        if between_loops == None: 
            between_loops = 3*loops 
        t = realign4d(self._runs, 
                      loops=loops,
                      between_loops=between_loops, 
                      speedup=speedup, 
                      optimizer=optimizer,
                      align_runs=align_runs, 
                      time_interp=self._time_interp,
                      affine_class=self.affine_class, 
                      metric=self.metric)
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
                 tr=None, tr_slices=None, start=0.0, time_interp=True, 
                 affine_class=Rigid, metric=METRIC):
        self._generic_init(images, affine_class, slice_order, interleaved, 
                           tr, tr_slices, start, time_interp, metric)

