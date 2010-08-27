# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Intensity-based matching. 

Questions: alexis.roche@gmail.com
"""

from constants import _OPTIMIZER, _XTOL, _FTOL, _GTOL, _STEP

from _registration import _joint_histogram, _similarity, builtin_similarities
from affine import Affine, apply_affine, inverse_affine, subgrid_affine
from grid_transform import GridTransform

from nipy.core.image.affine_image import AffineImage
from nipy.algorithms.optimization import fmin_steepest

import numpy as np  
from scipy.optimize import fmin as fmin_simplex, fmin_powell, fmin_cg, fmin_bfgs
from sys import maxint


_CLAMP_DTYPE = 'short' # do not edit
_BINS = 256
_INTERP = 'pv'
_OPTIMIZER = 'powell'

# Dictionary of interpolation methods
# pv: Partial volume 
# tri: Trilinear 
# rand: Random interpolation
interp_methods = {'pv': 0, 'tri': 1, 'rand': -1}


class IconicRegistration(object):
    """
    A class to reprensent a generic intensity-based image registration
    algorithm.
    """

    def __init__(self, source, target, bins=_BINS, source_mask=None, target_mask=None):

        """
        Creates a new iconic registration object.

        Parameters
        ----------
        source : nipy image-like
          Source image 

        target : nipy image-like
          Target image 
    
        bins : int or sequence of ints
          Number of histogram bins for each image

        source_mask : nipy image-like
          Mask to apply to source image 

        target_mask : nipy image-like
          Mask to apply to target image 

        """

        # Binning sizes  
        if not hasattr(bins, '__iter__'): 
            bins = [int(bins), int(bins)]

        # Source image binning
        mask = None
        if not source_mask == None: 
            mask = source_mask.get_data()
        data, s_bins = clamp(source.get_data(), bins=bins[0], mask=mask)
        self._source_image = AffineImage(data, source.affine, 'ijk')
        self.set_source_fov()
 
        # Target image binning and padding with -1 
        mask = None
        if not target_mask == None: 
            mask = target_mask.get_data()
        data, t_bins = clamp(target.get_data(), bins=bins[1], mask=mask)
        self._target = -np.ones(np.array(target.shape)+2, dtype=_CLAMP_DTYPE)
        self._target[1:-1, 1:-1, 1:-1] = data
        self._target_fromworld = inverse_affine(target.affine)
        
        # Histograms
        self._joint_hist = np.zeros([s_bins, t_bins])
        self._source_hist = np.zeros(s_bins)
        self._target_hist = np.zeros(t_bins)

        # Set default registration parameters
        self._set_interp()
        self._set_similarity()


    def _get_interp(self): 
        return interp_methods.keys()[interp_methods.values().index(self._interp)]
    
    def _set_interp(self, method=_INTERP): 
        self._interp = interp_methods[method]

    interp = property(_get_interp, _set_interp)
        
    def set_source_fov(self, spacing=[1,1,1], corner=[0,0,0], shape=None, 
                       fixed_npoints=None):
        
        if shape == None:
            shape = self._source_image.shape
            
        slicer = lambda : tuple([slice(corner[i],shape[i]+corner[i],spacing[i]) for i in range(3)])
        fov_data = self._source_image.get_data()[slicer()]

        # Adjust spacing to match desired number of points
        if fixed_npoints: 
            spacing = subsample(fov_data, npoints=fixed_npoints)
            fov_data = self._source_image.get_data()[slicer()]

        self._slices = slicer()
        self._source = fov_data
        self._source_npoints = (fov_data >= 0).sum()
        self._source_toworld = subgrid_affine(self._source_image.affine, self._slices)

    def _set_similarity(self, similarity='cr', pdf=None): 
        if isinstance(similarity, str): 
            self._similarity = builtin_similarities[similarity]
            self._similarity_func = None
        else: 
            # TODO: check that similarity is a function with the right
            # API: similarity(H) where H is the joint histogram 
            self._similarity = builtin_similarities['custom']
            self._similarity_func = similarity 

        ## Use array rather than asarray to ensure contiguity 
        self._pdf = np.array(pdf)  

    def _get_similarity(self):
        builtins = builtin_similarities.values()
        if self._similarity in builtins: 
            return builtin_similarities.keys()[builtins.index(self._similarity)]
        else: 
            return self._similarity_func

    similarity = property(_get_similarity, _set_similarity)

    def voxel_transform(self, T):
        """ 
        T is the 4x4 transformation between the real coordinate systems
        The corresponding voxel transformation is: Tv = Tt^-1 * T * Ts
        """
        ## C-contiguity required
        return np.dot(self._target_fromworld, np.dot(T, self._source_toworld)) 

    def eval(self, T):
        if isinstance(T, GridTransform): 
            # TODO: make sure T.shape matches self._source_image.shape
            affine = 0 
            Tv = apply_affine(self._target_fromworld, T[self._slices])
        else:
            affine = 1
            Tv = np.dot(self._target_fromworld, np.dot(T, self._source_toworld)) 
        seed = self._interp
        if self._interp < 0:
            seed = - np.random.randint(maxint)
        _joint_histogram(self._joint_hist, 
                         self._source.flat, ## array iterator
                         self._target, 
                         Tv,
                         affine,
                         seed)
        #self.source_hist = np.sum(self._joint_hist, 1)
        #self.target_hist = np.sum(self._joint_hist, 0)
        return _similarity(self._joint_hist, 
                           self._source_hist, 
                           self._target_hist, 
                           self._similarity, 
                           self._pdf, 
                           self._similarity_func)


    def optimize(self, start, method=_OPTIMIZER, **kwargs):

        T = start
        tc0 = T.param

        # Loss function to minimize
        def loss(tc):
            T.param = tc
            return -self.eval(T) 
    
        def callback(tc):
            T.param = tc
            print(T)
            print(str(self.similarity) + ' = %s' % self.eval(T))
            print('')
                  

        # Switching to the appropriate optimizer
        print('Initial guess...')
        print(T)
        if method=='powell':
            fmin = fmin_powell
            kwargs.setdefault('xtol', _XTOL)
            kwargs.setdefault('ftol', _FTOL)
        elif method=='steepest':
            fmin = fmin_steepest
            kwargs.setdefault('xtol', _XTOL)
            kwargs.setdefault('ftol', _FTOL)
            kwargs.setdefault('step', _STEP)
        elif method=='cg':
            fmin = fmin_cg
            kwargs.setdefault('gtol', _GTOL)
        elif method=='bfgs':
            fmin = fmin_bfgs
            kwargs.setdefault('gtol', _GTOL)
        else: # simplex method 
            fmin = fmin_simplex 
            kwargs.setdefault('xtol', _XTOL)
            kwargs.setdefault('ftol', _FTOL)
        
        # Output
        print ('Optimizing using %s' % fmin.__name__)
        T.param = fmin(loss, tc0, callback=callback, **kwargs)
        return T 


    def explore(self, T0, *args): 
    
        """
        Evaluate the similarity at the transformations specified by
        sequences of parameter values.

        For instance: 

        explore(T0, (0, [-1,0,1]), (4, [-2.,2]))
        """
        nparams = T0.param.size
        sizes = np.ones(nparams)
        deltas = [[0] for i in range(nparams)]
        for a in args:
            deltas[a[0]] = a[1]
        grids = np.mgrid[[slice(0, len(d)) for d in deltas]]
        ntrials = np.prod(grids.shape[1:])
        Deltas = [np.asarray(deltas[i])[grids[i,:]].ravel() for i in range(nparams)]
        simis = np.zeros(ntrials)
        params = np.zeros([nparams, ntrials])

        T = Affine()
        for i in range(ntrials):
            t = T0.param + np.array([D[i] for D in Deltas])
            T.param = t 
            simis[i] = self.eval(T)
            params[:, i] = t 

        return simis, params
        


def _clamp(x, y, bins=_BINS, mask=None):

    # Threshold
    dmaxmax = 2**(8*y.dtype.itemsize-1)-1
    dmax = bins-1 ## default output maximum value
    if dmax > dmaxmax: 
        raise ValueError('Excess number of bins')
    xmin = float(x.min())
    xmax = float(x.max())
    d = xmax-xmin

    """
    If the image dynamic is small, no need for compression: just
    downshift image values and re-estimate the dynamic range (hence
    xmax is translated to xmax-tth casted to the appropriate
    dtype. Otherwise, compress after downshifting image values (values
    equal to the threshold are reset to zero).
    """
    if issubclass(x.dtype.type, np.integer) and d<=dmax:
        y[:] = x-xmin
        bins = int(d)+1
    else: 
        a = dmax/d
        y[:] = np.round(a*(x-xmin))
 
    return y, bins 


def clamp(x, bins=_BINS, mask=None):
    """ 
    Clamp array values that fall within a given mask in the range
    [0..bins-1] and reset masked values to -1.
    
    Parameters
    ----------
    x : ndarray
      The input array

    bins : number 
      Desired number of bins

    mask : ndarray, tuple or slice
      Anything such that x[mask] is an array. 
    
    Returns
    -------
    y : ndarray
      Clamped array, masked items are assigned -1 

    bins : number 
      Adjusted number of bins 

    """
    y = -np.ones(x.shape, dtype=_CLAMP_DTYPE)
    if mask == None: 
        y, bins = _clamp(x, y, bins)
    else:
        xm = x[mask]
        ym = y[mask]
        ym, bins = _clamp(x, ym, bins)
        y[mask] = ym
    return y, bins


def subsample(data, npoints):
    """  
    Tune spacing factors so that the number of voxels in the
    output block matches a given number.
    
    Parameters
    ----------
    data : ndarray or sequence  
      Data image to subsample
    
    npoints : number
      Target number of voxels (negative values will be ignored)

    Returns
    -------
    spacing: ndarray 
      Spacing factors
                 
    """
    dims = data.shape
    actual_npoints = (data >= 0).sum()
    spacing = np.ones(3, dtype='uint')

    while actual_npoints > npoints:

        # Subsample the direction with the highest number of samples
        ddims = dims/spacing
        if ddims[0] >= ddims[1] and ddims[0] >= ddims[2]:
            dir = 0
        elif ddims[1] > ddims[0] and ddims[1] >= ddims[2]:
            dir = 1
        else:
            dir = 2
        spacing[dir] += 1
        subdata = data[::spacing[0], ::spacing[1], ::spacing[2]]
        actual_npoints = (subdata >= 0).sum()
            
    return spacing


