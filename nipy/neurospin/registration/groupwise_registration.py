# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from .constants import _OPTIMIZER, _XTOL, _FTOL, _GTOL, _STEP
from .affine import Rigid, Similarity, Affine, apply_affine
from ._cubic_spline import cspline_transform, cspline_sample3d, cspline_sample4d

from nipy.core.image.affine_image import AffineImage
from nipy.algorithms.optimize import fmin_steepest

import numpy as np
from scipy.optimize import fmin as fmin_simplex, fmin_powell, fmin_cg, fmin_bfgs

        
_SLICE_ORDER = 'ascending'
_INTERLEAVED = False
_SLICE_AXIS = 2 
_SPEEDUP = 4
_WITHIN_LOOPS = 2
_BETWEEN_LOOPS = 5 


def interp_slice_order(Z, slice_order): 
    Z = np.asarray(Z)
    nslices = len(slice_order)
    aux = np.asarray(list(slice_order)+[slice_order[0]+nslices])
    Zf = np.floor(Z).astype('int')
    w = Z - Zf
    Zal = Zf % nslices
    Za = Zal + w
    """
    Za = Z % nslices
    Zal = Za.astype('int')
    w = Za - Zal 
    """
    ret = (1-w)*aux[Zal] + w*aux[Zal+1]
    ret += (Z-Za)
    return ret


def grid_coords(xyz, affine, from_world, to_world):
    Tv = np.dot(from_world, np.dot(affine, to_world))
    XYZ = apply_affine(Tv, xyz)
    return XYZ[:,0], XYZ[:,1], XYZ[:,2]


class Image4d(object):
    """
    Class to represent a sequence of 3d scans acquired on a
    slice-by-slice basis.
    """
    def __init__(self, array, to_world, tr, tr_slices=None, start=0.0, 
                 slice_order=_SLICE_ORDER, interleaved=_INTERLEAVED, 
                 slice_axis=_SLICE_AXIS):
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
        self.to_world = to_world 
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
        ## assume that the world referential is 'scanner' as defined
        ## by the nifti norm
        self.reversed_slices = to_world[slice_axis][slice_axis]<0 

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


    def from_time(self, zv, t):
        """
        tv = from_time(zv, t)
        zv, tv are grid coordinates; t is an actual time value. 
        """
        corr = self.tr_slices*interp_slice_order(self.z_to_slice(zv), self.slice_order)
        return (t-self.start-corr)/self.tr



class Realign4d_Algorithm(object):

    def __init__(self, 
                 im4d, 
                 speedup=_SPEEDUP,
                 optimizer=_OPTIMIZER, 
                 affine_class=Rigid,
                 transforms=None, 
                 time_interp=True):
        self.optimizer = optimizer
        dims = im4d.array.shape
        self.dims = dims 
        self.nscans = dims[3]
        # Define mask
        speedup = max(1, int(speedup))
        xyz = np.mgrid[0:dims[0]:speedup, 0:dims[1]:speedup, 0:dims[2]:speedup]
        xyz = np.rollaxis(xyz, 0, 4)
        self.xyz = np.reshape(xyz, [np.prod(xyz.shape[0:-1]), 3])   
        masksize = self.xyz.shape[0]
        self.data = np.zeros([masksize, self.nscans], dtype='double')
        # Initialize space/time transformation parameters 
        self.to_world = im4d.to_world
        self.from_world = np.linalg.inv(self.to_world)
        if transforms == None: 
            self.transforms = [affine_class() for scan in range(self.nscans)]
        else: 
            self.transforms = transforms
        self.from_time = im4d.from_time
        self.timestamps = im4d.tr*np.arange(self.nscans)
        # Compute the 4d cubic spline transform
        self.time_interp = time_interp 
        if time_interp: 
            self.cbspline = cspline_transform(im4d.array)
        else: 
            self.cbspline = np.zeros(dims)
            for t in range(dims[3]): 
                self.cbspline[:,:,:,t] = cspline_transform(im4d.array[:,:,:,t])

    def resample_inmask(self, t):
        """
        x,y,z,t are "ideal grid" coordinates 
        X,Y,Z,T are "acquisition grid" coordinates 
        """
        X, Y, Z = grid_coords(self.xyz, self.transforms[t], 
                              self.from_world, self.to_world)
        if self.time_interp: 
            T = self.from_time(Z, self.timestamps[t])
            cspline_sample4d(self.data[:,t], self.cbspline, X, Y, Z, T)
        else: 
            cspline_sample3d(self.data[:,t], self.cbspline[:,:,:,t], X, Y, Z)

    def resample_all_inmask(self):
        for t in range(self.nscans):
            print('Resampling scan %d/%d' % (t+1, self.nscans))
            self.resample_inmask(t)

    def init_motion_detection(self, t):
        """
        The idea is to compute the global variance using the following
        decomposition:

        V = (n-1)/n V1 + (n-1)/n^2 (x1-m1)^2
          = alpha + beta d2,

        with alpha=(n-1)/n V1, beta = (n-1)/n^2, d2 = (x1-m1)^2. 
        
        Only the second term is variable when one image moves while
        all other images are fixed.
        """
        self.resample_inmask(t)
        fixed = range(self.nscans)
        fixed.remove(t)
        aux = self.data[:, fixed]
        self.m1 = aux.mean(1)
        self.d2 = np.zeros(np.shape(self.m1))
        self.alpha = ((self.nscans-1.0)/self.nscans)*aux.var(1).mean()
        self.beta = (self.nscans-1.0)/self.nscans**2
            
    def msid(self, t):
        """
        Mean square intensity difference
        """
        self.resample_inmask(t)
        self.d2[:] = self.data[:,t]
        self.d2 -= self.m1
        self.d2 **= 2
        return self.d2.mean()

    def variance(self, t): 
        return self.alpha + self.beta*self.msid(t)

    def safe_variance(self, t):
        """
        No need to invoke self.init_motion_detection.
        """
        self.resample_inmask(t)
        self.m = self.data.mean(1)
        self.m2 = (self.data**2).mean(1)
        self.m **= 2
        self.m2 -= self.m
        return self.m2.mean()

    def estimate_motion(self):
        optimizer = self.optimizer

        def callback(pc):
            self.transforms[t].param = pc
            print(self.transforms[t])

        if optimizer=='powell':
            tols = {'xtol': _XTOL, 'ftol': _FTOL}
            fmin = fmin_powell
        elif optimizer=='steepest':
            tols = {'xtol': _XTOL, 'ftol': _FTOL, 'step':_STEP}
            fmin = fmin_steepest
        elif optimizer=='cg':
            tols = {'gtol': _GTOL}
            fmin = fmin_cg
        elif optimizer=='bfgs':
            tols = {'gtol': _GTOL}
            fmin = fmin_bfgs
        else: # simplex method 
            tols = {'xtol': _XTOL, 'ftol': _FTOL}
            fmin = fmin_simplex

        # Resample data according to the current space/time transformation 
        self.resample_all_inmask()

        # Optimize motion parameters 
        for t in range(self.nscans):
            print('Correcting motion of scan %d/%d...' % (t+1, self.nscans))
            def cost(pc):
                self.transforms[t].param = pc
                return self.msid(t)
            self.init_motion_detection(t)
            self.transforms[t].param = fmin(cost, self.transforms[t].param,
                                            callback=callback, **tols)

        # At this stage, transforms map an implicit 'ideal' grid to
        # the 'acquisition' grid. We redefine the ideal grid as being
        # conventionally aligned with the first scan.
        T0inv = self.transforms[0].inv()
        for t in range(self.nscans): 
            self.transforms[t] = self.transforms[t]*T0inv 
        


    def resample(self):
        print('Gridding...')
        dims = self.dims
        XYZ = np.mgrid[0:dims[0], 0:dims[1], 0:dims[2]]
        XYZ = np.rollaxis(XYZ, 0, 4)
        XYZ = np.reshape(XYZ, [np.prod(XYZ.shape[0:-1]), 3])
        res = np.zeros(dims)
        for t in range(self.nscans):
            print('Fully resampling scan %d/%d' % (t+1, self.nscans))
            X, Y, Z = grid_coords(XYZ, self.transforms[t], 
                                  self.from_world, self.to_world)
            if self.time_interp: 
                T = self.from_time(Z, self.timestamps[t])
                cspline_sample4d(res[:,:,:,t], self.cbspline, X, Y, Z, T)
            else: 
                cspline_sample3d(res[:,:,:,t], self.cbspline[:,:,:,t], X, Y, Z)
        return res
    


def resample4d(im4d, transforms, time_interp=True): 
    """
    corr_im4d_array = resample4d(im4d, transforms=None, time_interp=True)
    """
    r = Realign4d_Algorithm(im4d, transforms=transforms, time_interp=time_interp)
    return r.resample()



def single_run_realign4d(im4d, 
                         loops=_WITHIN_LOOPS, 
                         speedup=_SPEEDUP, 
                         optimizer=_OPTIMIZER, 
                         affine_class=Rigid, 
                         time_interp=True): 
    """
    transforms = single_run_realign4d(im4d, loops=2, speedup=4, optimizer='powell', time_interp=True)

    Parameters
    ----------
    im4d : Image4d instance

    """ 
    r = Realign4d_Algorithm(im4d, speedup=speedup, optimizer=optimizer, 
                            time_interp=time_interp, affine_class=affine_class)
    for loop in range(loops): 
        r.estimate_motion()
    return r.transforms

def realign4d(runs, 
              within_loops=_WITHIN_LOOPS, 
              between_loops=_BETWEEN_LOOPS, 
              speedup=_SPEEDUP, 
              optimizer=_OPTIMIZER, 
              align_runs=True, 
              time_interp=True, 
              affine_class=Rigid): 
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
    transforms = [single_run_realign4d(run, loops=within_loops, 
                                       speedup=speedup, optimizer=optimizer,
                                       time_interp=time_interp, 
                                       affine_class=affine_class) for run in runs]
    if not align_runs: 
        return transforms, transforms, None

    # Correct between-session motion using the mean image of each corrected run 
    corr_runs = [resample4d(runs[i], transforms=transforms[i], time_interp=time_interp) for i in range(nruns)]
    aux = np.rollaxis(np.asarray([c.mean(3) for c in corr_runs]), 0, 4)
    ## Fake time series with zero inter-slice time 
    ## FIXME: check that all runs have the same to-world transform
    mean_img = Image4d(aux, to_world=runs[0].to_world, tr=1.0, tr_slices=0.0) 
    transfo_mean = single_run_realign4d(mean_img, loops=between_loops, speedup=speedup, 
                                        optimizer=optimizer, time_interp=time_interp)

    # Compose transformations for each run
    ctransforms = [None for i in range(nruns)]
    for i in range(nruns):
        ctransforms[i] = [t*transfo_mean[i] for t in transforms[i]]
    return ctransforms, transforms, transfo_mean


def split_affine(a): 
    sa = np.eye(4)
    sa[0:3, 0:3] = a[0:3, 0:3]
    return sa, a[3,3]


class Realign4d(object): 

    def __init__(self, images, affine_class=Rigid):
        self._generic_init(images, affine_class, _SLICE_ORDER, _INTERLEAVED, 1.0, 0.0, 0.0, False)

    def _generic_init(self, images, affine_class, 
                      slice_order, interleaved, tr, tr_slices, start, time_interp):
        if not hasattr(images, '__iter__'):
            images = [images]
        self._runs = []
        self.affine_class = affine_class
        for im in images: 
            spatial_affine, _tr = split_affine(im.affine)
            if tr == None: 
                tr = _tr
            self._runs.append(Image4d(im.get_data(), spatial_affine, tr=tr, tr_slices=tr_slices, 
                                      start=start, slice_order=slice_order, interleaved=interleaved)) 
        self._transforms = [None for run in self._runs]
        self._time_interp = time_interp 
                      
    def estimate(self, iterations=2, between_loops=None, align_runs=True): 
        within_loops = iterations 
        if between_loops == None: 
            between_loops = 3*within_loops 
        t = realign4d(self._runs, within_loops=within_loops, 
                      between_loops=between_loops, align_runs=align_runs, 
                      time_interp=self._time_interp, affine_class=self.affine_class)
        self._transforms, self._within_run_transforms, self._mean_transforms = t

    def resample(self, align_runs=True): 
        """
        Return a list of 4d nibabel-like images corresponding to the resampled runs. 
        """
        if align_runs: 
            transforms = self._transforms
        else: 
            transforms = self._within_run_transforms
        runs = range(len(self._runs))
        data = [resample4d(self._runs[r], transforms=transforms[r], time_interp=self._time_interp) for r in runs]
        return [AffineImage(data[r], self._runs[r].to_world, 'ijk') for r in runs]



class FmriRealign4d(Realign4d): 

    def __init__(self, images, slice_order, interleaved,
                 tr=None, tr_slices=None, start=0.0, time_interp=True, 
                 affine_class=Rigid):
        self._generic_init(images, affine_class, slice_order, interleaved, 
                           tr, tr_slices, start, time_interp)

