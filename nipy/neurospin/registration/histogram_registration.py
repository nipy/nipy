# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Intensity-based matching. 

Questions: alexis.roche@gmail.com
"""

from .constants import _OPTIMIZER, _XTOL, _FTOL, _GTOL, _STEP

from ._registration import _joint_histogram, _similarity, builtin_similarities
from .affine import Affine, apply_affine, inverse_affine, subgrid_affine
from .grid_transform import GridTransform

from nipy.core.image.affine_image import AffineImage
from nipy.algorithms.optimize import fmin_steepest

import numpy as np  
from scipy.optimize import fmin as fmin_simplex, fmin_powell, fmin_cg, fmin_bfgs
from sys import maxint


_CLAMP_DTYPE = 'short' # do not edit
_BINS = 256
_INTERP = 'pv'
_NPOINTS = 64**3

# Dictionary of interpolation methods
# pv: Partial volume 
# tri: Trilinear 
# rand: Random interpolation
interp_methods = {'pv': 0, 'tri': 1, 'rand': -1}


class HistogramRegistration(object):
    """
    A class to reprensent a generic intensity-based image registration
    algorithm.
    """

    def __init__(self, from_img, to_img, from_bins=_BINS, to_bins=None, 
                 from_mask=None, to_mask=None):

        """
        Creates a new histogram registration object.

        Parameters
        ----------
        from_img : nipy image-like
          `From` image 

        to_img : nipy image-like
          `To` image 
    
        from_bins : integer
          Number of histogram bins to represent the `from` image

        to_bins : integer
          Number of histogram bins to represent the `to` image

        from_mask : nipy image-like
          Mask to apply to the `from` image 

        to_mask : nipy image-like
          Mask to apply to the `to` image 

        """

        # Binning sizes  
        if to_bins == None: 
            to_bins = from_bins 

        # Clamping of the `from` image. The number of bins may be
        # overriden if unnecessarily large. 
        mask = None
        if not from_mask == None: 
            mask = from_mask.get_data()
        data, from_bins = clamp(from_img.get_data(), bins=from_bins, mask=mask)
        self._from_img = AffineImage(data, from_img.affine, 'ijk')
        self.subsample()
        # self._from_world_coords is a Nx3 array representing the world
        # coordinates of the `from` clamped (subsampled) image voxels 
        tmp = np.mgrid[[slice(0, s) for s in self._from_data.shape]]
        tmp = np.rollaxis(tmp, 0, 1+self._from_data.ndim)
        self._from_world_coords = apply_affine(self._from_affine, tmp)

        # Clamping of the `to` image including padding with -1 
        mask = None
        if not to_mask == None: 
            mask = to_mask.get_data()
        data, to_bins = clamp(to_img.get_data(), bins=to_bins, mask=mask)
        self._to_data = -np.ones(np.array(to_img.shape)+2, dtype=_CLAMP_DTYPE)
        self._to_data[1:-1, 1:-1, 1:-1] = data
        self._to_inv_affine = inverse_affine(to_img.affine)
        
        # Histograms
        self._joint_hist = np.zeros([from_bins, to_bins])
        self._from_hist = np.zeros(from_bins)
        self._to_hist = np.zeros(to_bins)

        # Set default registration parameters
        self._set_interp()
        self._set_similarity()


    def _get_interp(self): 
        return interp_methods.keys()[interp_methods.values().index(self._interp)]
    
    def _set_interp(self, method=_INTERP): 
        self._interp = interp_methods[method]

    interp = property(_get_interp, _set_interp)
        
    def subsample(self, spacing=None, corner=[0,0,0], size=None, npoints=_NPOINTS):
        """ 
        Defines a subset of the `from` image to restrict joint
        histogram computation.

        Parameters
        ----------
        
        spacing : sequence (3,) of positive integers
          Subsampling factors 

        corner : sequence (3,) of positive integers
          Bounding box origin in voxel coordinates

        size : sequence (3,) of positive integers
          Desired bounding box size 

        npoints : positive integer
          Desired number of voxels in the bounding box. If a `spacing`
          argument is provided, then `npoints` is ignored.
        """
        if spacing == None: 
            spacing = [1,1,1]
        else: 
            npoints = None

        if size == None:
            size = self._from_img.shape
            
        slicer = lambda : tuple([slice(corner[i],size[i]+corner[i],spacing[i]) for i in range(3)])
        fov_data = self._from_img.get_data()[slicer()]

        # Adjust spacing to match desired field of view size
        if npoints: 
            spacing = ideal_spacing(fov_data, npoints=npoints)
            fov_data = self._from_img.get_data()[slicer()]

        self._from_data = fov_data
        self._from_npoints = (fov_data >= 0).sum()
        self._from_affine = subgrid_affine(self._from_img.affine, slicer())

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
        The corresponding voxel transformation is: Tt^-1 * T * Ts
        """
        return np.dot(self._to_inv_affine, np.dot(T, self._from_affine)) 

    def eval(self, T):
        # trans_voxel_coords needs be C-contiguous
        trans_voxel_coords = apply_affine(self._to_inv_affine, T(self._from_world_coords))
        interp = self._interp
        if self._interp < 0:
            interp = - np.random.randint(maxint)
        _joint_histogram(self._joint_hist, 
                         self._from_data.flat, ## array iterator
                         self._to_data, 
                         trans_voxel_coords,
                         interp)
        return _similarity(self._joint_hist, 
                           self._from_hist, 
                           self._to_hist, 
                           self._similarity, 
                           self._pdf, 
                           self._similarity_func)


    def optimize(self, start, method=_OPTIMIZER, **kwargs):

        T = start
        tc0 = T.param

        # Cost function to minimize
        def cost(tc):
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
        T.param = fmin(cost, tc0, callback=callback, **kwargs)
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
    if bins > np.iinfo(np.short).max:
        raise ValueError('Too large a bin size')
    y = -np.ones(x.shape, dtype=_CLAMP_DTYPE)
    if mask == None: 
        y, bins = _clamp(x, y, bins)
    else:
        xm = x[mask]
        ym = y[mask]
        ym, bins = _clamp(x, ym, bins)
        y[mask] = ym
    return y, bins


def ideal_spacing(data, npoints):
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


