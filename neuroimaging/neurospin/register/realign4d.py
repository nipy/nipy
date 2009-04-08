from routines import cspline_transform, cspline_sample4d
from transform import Affine, apply, BRAIN_RADIUS_MM

import numpy as np
import scipy as sp
import scipy.optimize
        

def grid_coords(xyz, affine, fromworld, toworld):
    Tv = np.dot(fromworld, np.dot(affine, toworld))
    XYZ = affine_transform.apply(xyz, Tv)
    return XYZ[0,:], XYZ[1,:], XYZ[2,:]



class TimeSeries:
    """
    Local class for code readability. Not to be exposed. 
    """
    def __init__(self, array, toworld, tr, tr_slices=None, start=0.0, 
                 slice_axis=2, reversed_slices=False, 
                 slice_order='ascending', interleaved=False):
        """
        Configure fMRI acquisition time parameters.
        
        tr  : inter-scan repetition time, i.e. the time elapsed between two consecutive scans
        tr_slices : inter-slice repetition time, same as tr for slices
        start   : starting acquisition time respective to the implicit time origin
        slice_order : string or array 
        """
        self.array = array 
        self.toworld = toworld 
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
        self.reversed_slices = bool(reversed_slices)

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

    def totime(self, z, t):
        return(self.start + self.tr*t + slice_time(self.z_to_slice(z), self.tr_slices, self.slice_order))

    def fromtime(self, z, t):
        return((t - self.start - slice_time(self.z_to_slice(z), self.tr_slices, self.slice_order))/self.tr)


class Realign4d:

    def __init__(self, time_series, speedup=4, optimizer='powell', transforms=None):
        self.optimizer = optimizer
        dims = time_series.array.shape
        self.dims = dims 
        self.nscans = dims[3]
        # Define mask
        speedup = max(1, int(speedup))
        xyz = np.mgrid[0:dims[0]:speedup, 0:dims[1]:speedup, 0:dims[2]:speedup]
        self.xyz = xyz.reshape(3, np.prod(xyz.shape[1::]))   
        masksize = self.xyz.shape[1]
        self.data = np.zeros([masksize, self.nscans], dtype='double')
        # Initialize space/time transformation parameters 
        self.toworld = time_series.toworld
        self.fromworld = np.linalg.inv(self.toworld)
        if transforms == None: 
            self.transforms = [Affine('rigid', radius=brain_radius_mm) for scan in range(self.nscans)]
        else: 
            self.transforms = transforms
        self.fromtime = time_series.fromtime
        self.timestamps = time_series.tr*np.array(range(self.nscans))
        # Compute the 4d cubic spline transform
        self.cbspline = cspline_transform(time_series.array)
              
    def resample_inmask(self, t):
        X, Y, Z = grid_coords(self.xyz, self.transforms[t], self.fromworld, self.toworld)
        T = self.fromtime(Z, self.timestamps[t])
        cspline_sample4d(self.data[:,t], self.cbspline, X, Y, Z, T)

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

    def correct_motion(self):
        optimizer = self.optimizer

        def callback(pc):
            p = pc*precond
            print('')
            print('  translation : %s' % p[0:3].__str__())
            print('  rotation    : %s' % p[3:6].__str__())
            print('')

        if optimizer=='simplex':
            fmin = sp.optimize.fmin
        elif optimizer=='powell':
            fmin = sp.optimize.fmin_powell
        elif optimizer=='conjugate gradient':
            fmin = sp.optimize.fmin_cg
        else:
            raise ValueError('Unrecognized optimizer')

        # Resample data according to the current space/time transformation 
        self.resample_all_inmask()

        # Optimize motion parameters 
        for t in range(self.nscans):
            print('Correcting motion of scan %d/%d...' % (t+1, self.nscans))

            def loss(pc):
                self.transforms[t].from_param(pc)
                return self.msid(t)
        
            self.init_motion_detection(t)
            pc0 = self.transforms[t].to_param()
            pc = fmin(loss, pc0, callback=callback)
            self.transforms[t].from_param(pc)


    def resample(self):
        print('Gridding...')
        dims = self.dims
        XYZ = np.mgrid[0:dims[0], 0:dims[1], 0:dims[2]]
        XYZ = XYZ.reshape(3, np.prod(XYZ.shape[1::]))
        res = np.zeros(dims)
        for t in range(self.nscans):
            print('Fully resampling scan %d/%d' % (t+1, self.nscans))
            X, Y, Z = grid_coords(XYZ, self.transforms[t], self.fromworld, self.toworld)
            T = self.fromtime(Z, self.timestamps[t])
            cspline_sample4d(res[:,:,:,t], self.cbspline, X, Y, Z, T)
        return res
    




def resample4d(time_series, transforms=None): 
    """
    corr_time_series = resample4d(time_series, transforms=None)
    """
    r = Realign4d(time_series, transforms=transforms)
    return r.resample()


def _realign4d(time_series, loops=2, speedup=4, optimizer='powell'): 
    """
    transforms = _realign4d(time_series, loops=2, speedup=4, optimizer='powell')

    Parameters
    ----------
    time_series : TimeSeries instance

    """ 
    r = Realign4d(time_series, speedup=speedup, optimizer=optimizer)
    for loop in range(loops): 
        r.correct_motion()
    return r.transforms

def realign4d(runs, within_loops=2, between_loops=5, speedup=4, optimizer='powell'): 
    """
    transforms = realign4d(runs, within_loops=2, bewteen_loops=5, speedup=4, optimizer='powell')

    Parameters
    ----------

    runs : list of TimeSeries objects
    
    Returns
    -------
    transforms : list
                 nested list of rigid transformations
    """

    # Single-session case
    if not isinstance(runs, list): 
        runs = [runs]
    
    # Correct motion and slice timing in each sequence separately
    transfo_runs = [_realign4d(run, loops=within_loops, speedup=speedup, optimizer=optimizer) for run in runs]
    if len(runs)==1: 
        return transfo_runs[0]

    # Correct between-session motion using the mean image of each corrected run 
    corr_run = [resample4d(run, transforms=transfo_runs) for run in runs]
    aux = np.rollaxis(np.asarray([corr_run.array.mean(3) for corr_run in corr_runs]), 0, 4)
    ## fake time series using the first run's to-world transform
    ## FIXME: check that all runs have the same to-world transform
    mean_img = TimeSeries(aux, toworld=run[0].toworld, tr_slices=0.0) 
    transfo_mean = _realign4d(mean_img, loops=between_loops, speedup=speedup, optimizer=optimizer)
    corr_mean = resample4d(mean_img, transforms=transfo_mean)
    
    # Compose transformations for each run
    for idx in range(len(runs)):
        sess_toworld = transfo_mean[idx]
        transforms = [sess_toworld*T for T in transfo_runs[idx]]
        transfo_runs[idx] = transforms
        
    return transfo_runs



