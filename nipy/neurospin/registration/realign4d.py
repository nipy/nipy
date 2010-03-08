from affine import Rigid

from nipy.neurospin.image import apply_affine
from nipy.neurospin.image.image_module import cspline_transform, cspline_sample4d
from nipy.neurospin.utils.optimize import fmin_steepest

import numpy as np
from scipy.optimize import fmin as fmin_simplex, fmin_powell, fmin_cg, fmin_bfgs
        
_speedup = 4
_optimizer = 'powell'
_within_loops = 2
_between_loops = 5 

_xtol = .1
_ftol = .01
_gtol = .1 



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
                 slice_order='ascending', interleaved=False, slice_axis=2):
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



class Realign4d(object):

    def __init__(self, 
                 im4d, 
                 speedup=_speedup,
                 optimizer=_optimizer, 
                 transforms=None):
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
            self.transforms = [Rigid() for scan in np.arange(self.nscans)]
        else: 
            self.transforms = transforms
        self.from_time = im4d.from_time
        self.timestamps = im4d.tr*np.arange(self.nscans)
        # Compute the 4d cubic spline transform
        self.cbspline = cspline_transform(im4d.array)
              
    def resample_inmask(self, t):
        X, Y, Z = grid_coords(self.xyz, self.transforms[t], 
                              self.from_world, self.to_world)
        T = self.from_time(Z, self.timestamps[t])
        cspline_sample4d(self.data[:,t], self.cbspline, X, Y, Z, T)

    def resample_all_inmask(self):
        for t in np.arange(self.nscans):
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

    def correct_motion(self):
        optimizer = self.optimizer

        def callback(pc):
            self.transforms[t].param = pc
            print(self.transforms[t])

        if optimizer=='powell':
            tols = {'xtol': _xtol, 'ftol': _ftol}
            fmin = fmin_powell
        elif optimizer=='steepest':
            tols = {'xtol': _xtol, 'ftol': _ftol, 'epsilon':_epsilon}
            fmin = fmin_steepest
        elif optimizer=='cg':
            tols = {'gtol': _gtol}
            fmin = fmin_cg
        elif optimizer=='bfgs':
            tols = {'gtol': _gtol}
            fmin = fmin_bfgs
        else: # simplex method 
            tols = {'xtol': _xtol, 'ftol': _ftol}
            fmin = fmin_simplex

        # Resample data according to the current space/time transformation 
        self.resample_all_inmask()

        # Optimize motion parameters 
        for t in np.arange(self.nscans):
            print('Correcting motion of scan %d/%d...' % (t+1, self.nscans))
            def loss(pc):
                self.transforms[t].param = pc
                return self.msid(t)
            self.init_motion_detection(t)
            self.transforms[t].param = fmin(loss, self.transforms[t].param,
                                            callback=callback, **tols)

        # At this stage, transforms map an implicit 'ideal' grid to
        # the 'acquisition' grid. We redefine the ideal grid as being
        # conventionally aligned with the first scan.
        T0inv = self.transforms[0].inv()
        for t in np.arange(self.nscans): 
            self.transforms[t] = self.transforms[t]*T0inv 
        


    def resample(self):
        print('Gridding...')
        dims = self.dims
        XYZ = np.mgrid[0:dims[0], 0:dims[1], 0:dims[2]]
        XYZ = np.rollaxis(XYZ, 0, 4)
        XYZ = np.reshape(XYZ, [np.prod(XYZ.shape[0:-1]), 3])
        res = np.zeros(dims)
        for t in np.arange(self.nscans):
            print('Fully resampling scan %d/%d' % (t+1, self.nscans))
            X, Y, Z = grid_coords(XYZ, self.transforms[t], 
                                  self.from_world, self.to_world)
            T = self.from_time(Z, self.timestamps[t])
            cspline_sample4d(res[:,:,:,t], self.cbspline, X, Y, Z, T)
        return res
    




def resample4d(im4d, transforms=None): 
    """
    corr_im4d_array = resample4d(im4d, transforms=None)
    """
    r = Realign4d(im4d, transforms=transforms)
    return r.resample()



def _realign4d(im4d, 
               loops=_within_loops, 
               speedup=_speedup, 
               optimizer=_optimizer): 
    """
    transforms = _realign4d(im4d, loops=2, speedup=4, optimizer='powell')

    Parameters
    ----------
    im4d : Image4d instance

    """ 
    r = Realign4d(im4d, speedup=speedup, optimizer=optimizer)
    for loop in np.arange(loops): 
        r.correct_motion()
    return r.transforms

def realign4d(runs, 
              within_loops=_within_loops, 
              between_loops=_between_loops, 
              speedup=_speedup, 
              optimizer=_optimizer, 
              align_runs=True): 
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
    transforms = [_realign4d(run, loops=within_loops, 
                             speedup=speedup, optimizer=optimizer) for run in runs]
    if not align_runs: 
        return transforms, transforms, None

    # Correct between-session motion using the mean image of each corrected run 
    corr_runs = [resample4d(runs[i], transforms=transforms[i]) for i in np.arange(nruns)]
    aux = np.rollaxis(np.asarray([c.mean(3) for c in corr_runs]), 0, 4)
    ## Fake time series with zero inter-slice time 
    ## FIXME: check that all runs have the same to-world transform
    mean_img = Image4d(aux, to_world=runs[0].to_world, tr=1.0, tr_slices=0.0) 
    transfo_mean = _realign4d(mean_img, loops=between_loops, speedup=speedup, 
                              optimizer=optimizer)

    # Compose transformations for each run
    ctransforms = []
    for i in np.arange(nruns):
        ctransforms[i] = [t*transfo_mean[i] for t in transforms[i]]
    return ctransforms, transforms, transfo_mean



