from routines import cspline_transform, cspline_sample4d

import affine_transform 

import numpy as np
import scipy as sp
import scipy.optimize
        
RADIUS_MM = 10 
RIGID = affine_transform.affine_types['3d']['rigid']


def grid_coords(xyz, params, r2v, v2r, transform=None):
    T = affine_transform.matrix44(params)
    Tv = np.dot(r2v, np.dot(T, v2r))
    XYZ = affine_transform.apply(xyz, Tv)
    return XYZ[0,:], XYZ[1,:], XYZ[2,:]



class TimeTransform:

    def __init__(self, nslices, tr, tr_slices=None, start=0.0, 
                 slice_axis=2, reversed_slices=False, slice_order='ascending', interleaved=False):
        """
        Configure fMRI acquisition time parameters.
        
        tr  : inter-scan repetition time, i.e. the time elapsed between two consecutive scans
        tr_slices : inter-slice repetition time, same as tr for slices
        start   : starting acquisition time respective to the implicit time origin
        slice_order : string or array 
        """
        
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
        self.nslices = int(nslices)
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

    def __call__(self, z, t):
        return(self.start + self.tr*t + slice_time(self.z_to_slice(z), self.tr_slices, self.slice_order))


    def inverse(self, z, t):
        return((t - self.start - slice_time(self.z_to_slice(z), self.tr_slices, self.slice_order))/self.tr)


class Realign4d:

    def __init__(self, img, transform, time_transform, speedup=4, optimizer='powell'):
        self.speedup = speedup
        self.optimizer = optimizer
        dims = img.shape
        self.time_transform = time_transform
        self.dims = dims 
        self.nscans = self.dims[3]
        # Define mask
        speedup = max(1, int(speedup))
        xyz = np.mgrid[0:dims[0]:speedup, 0:dims[1]:speedup, 0:dims[2]:speedup]
        self.xyz = xyz.reshape(3, np.prod(xyz.shape[1::]))   
        self.masksize = self.xyz.shape[1]
        self.data = np.zeros([self.masksize, self.nscans], dtype='double')
        # Initialize space/time transformation parameters 
        self.v2r = transform
        self.r2v = np.linalg.inv(transform)
        self.space_params = np.zeros([self.nscans, 6])
        self.time_params = img.tr*np.array(range(self.nscans))
        # Compute the 4d cubic spline transform
        self.cbspline = cspline_transform(img)
              
    def resample_inmask(self, t):
        X, Y, Z = grid_coords(self.xyz, self.space_params[t,:], self.r2v, self.v2r)
        T = self.time_transform.inverse(Z, self.time_params[t])
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
        precond = affine_transform.preconditioner(RADIUS_MM)[0:6]

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
            raise ValueError, 'Unrecognized optimizer.'

        # Resample data according to the current space/time transformation 
        self.resample_all_inmask()

        # Optimize motion parameters 
        for t in range(self.nscans):
            print('Correcting motion of scan %d/%d...' % (t+1, self.nscans))
           
            def loss(pc):
                p = pc*precond
                self.space_params[t,:] = p
                return self.msid(t)
        
            self.init_motion_detection(t)
            pc0 = self.space_params[t,:]/precond
            pc = fmin(loss, pc0, callback=callback)
            self.space_params[t,:] = pc*precond


    def resample(self, transforms=None):
        print('Gridding...')
        dims = self.dims
        XYZ = np.mgrid[0:dims[0], 0:dims[1], 0:dims[2]]
        XYZ = XYZ.reshape(3, np.prod(XYZ.shape[1::]))
        res = np.zeros(dims)
        for t in range(self.nscans):
            print('Fully resampling scan %d/%d' % (t+1, self.nscans))
            if transforms == None: 
                transform = None
            else:
                transform = transforms[t,:]
            X, Y, Z = grid_coords(XYZ, self.space_params[t,:], self.r2v, self.v2r, transform=transform)
            T = self.time_transform.inverse(Z, self.time_params[t])
            cspline_sample4d(res[:,:,:,t], self.cbspline, X, Y, Z, T)
        return res
    


def _realign4d(img, transform, time_transform, loops=2, speedup=4, optimizer='powell'): 
    """

    Parameters
    ----------
    img : ndarray
          4d image data

    transform : ndarray 
                Voxel-to-world affine (4x4 matrix)

    """ 
    r = Realign4d(img, transform, time_transform, speedup=speedup, optimizer=optimizer)
    for loop in range(loops): 
        r.correct_motion()
    return r.resample(), r.space_params



def _resample4d(img, transforms=None): 
    """
    corr_img, transfo_img = realign4d(runs, loops=2, speedup=4, optimizer='powell')

    Assumes img is a fff2.neuro.fmri_image instance. 
    """
    if not isinstance(img, fff2.neuro.fmri_image):
        raise ValueError, 'Wrong input object type: ' + str(type(img))

    r = Realign4d(img)
    corr_img = fff2.neuro.fmri_image(fff2.neuro.image(img), tr=img.tr, tr_slices=0.0)
    corr_img.set_array(r.resample(transforms=transforms))
    return corr_img


def params_to_mat44(transfo_run, transform=None):
    if transform == None:
        transform = np.eye(4)
    transforms = []
    for t in range(transfo_run.shape[0]):
        T = np.dot(transform, affine_transform.matrix44(transfo_run[t,:]))
        transforms.append(T)
    transforms = np.asarray(transforms)
    return transforms


def realign4d(runs, within_loops=2, between_loops=5, speedup=4, optimizer='powell'): 
    """
    corr_runs, transforms = realign4d(runs, within_loops=2, bewteen_loops=5, speedup=4, optimizer='powell')

    Assumes runs is a list of fff2.neuro.fmri_image instance. 
    """

    # Single-session case
    if not isinstance(runs, list): 
        corr_run, transfo_run = _realign4d(runs, loops=within_loops, speedup=speedup, optimizer=optimizer)
        return corr_run, params_to_mat44(transfo_run)
    
    # Correct motion and slice timing in each sequence separately
    corr_runs = []
    transfo_runs = []
    for run in runs:
        corr_run, transfo_run = _realign4d(run, loops=within_loops, speedup=speedup, optimizer=optimizer)
        corr_runs.append(corr_run)
        transfo_runs.append(transfo_run)

    # Create a pseudo fmri_image using the means of each corrected run 
    aux = []
    run_idx = range(len(runs))
    for idx in run_idx:
        run = corr_runs[idx]
        aux.append(run.array.mean(3))
    aux = np.rollaxis(np.asarray(aux), 0, 4)
    mean_img = fff2.neuro.fmri_image(fff2.neuro.image(aux, transform=run.transform), tr_slices=0.0)
    corr_mean, transfo_mean = _realign4d(mean_img, loops=between_loops, speedup=speedup, optimizer=optimizer)
    
    # Compose transformations for each run
    for idx in run_idx:
        run = corr_runs[idx]
        transforms = params_to_mat44(transfo_runs[idx], transform=affine_transform.matrix44(transfo_mean[idx]))
        corr_runs[idx] = _resample4d(run, transforms=transforms)
        transfo_runs[idx] = transforms
        
    return corr_runs, transfo_runs, corr_mean, transfo_mean 



