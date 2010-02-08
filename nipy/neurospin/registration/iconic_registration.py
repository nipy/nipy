"""
Intensity-based matching. 

Questions: alexis.roche@gmail.com
"""
from nipy.neurospin.image import Image, set_image
from registration_module import _joint_histogram, _similarity, builtin_similarities
from affine import Affine

import numpy as np  
from scipy.optimize import fmin, fmin_powell, fmin_cg
from sys import maxint

# Globals
_clamp_dtype = 'short'


# Dictionary of interpolation methods
# pv: Partial volume 
# tri: Trilinear 
# rand: Random interpolation
interp_methods = {'pv': 0, 'tri': 1, 'rand': -1}


class IconicRegistration(object):

    def __init__(self, source, target, bins = 256):

        """
        A class to reprensent a generic intensity-based image
        registration algorithm.
        """

        # Binning size  
        if isinstance (bins, (int, long, float)): 
            bins = [int(bins), int(bins)]

        # Source image binning
        values, s_bins = clamp(source(), bins=bins[0])
        self._source_image = set_image(source, values)
        self.set_source_fov()
 
        # Target image padding + binning
        values, t_bins = clamp(target(), bins=bins[1])
        _target_image = set_image(target, values)
        self._target = -np.ones(np.array(target.shape)+2, dtype=_clamp_dtype)
        _view = self._target[1:-1, 1:-1, 1:-1]
        _view[:] = _target_image.data[:]
        self._target_fromworld = target.inv_affine
        
        # Histograms
        self._joint_hist = np.zeros([s_bins, t_bins])
        self._source_hist = np.zeros(s_bins)
        self._target_hist = np.zeros(t_bins)

        # Set default registration parameters
        self._set_interp()
        self._set_similarity()


    def _get_interp(self): 
        return interp_methods.keys()[interp_methods.values().index(self._interp)]
    
    def _set_interp(self, method='pv'): 
        self._interp = interp_methods[method]

    interp = property(_get_interp, _set_interp)
        
    def set_source_fov(self, spacing=[1,1,1], corner=[0,0,0], shape=None, 
                       fixed_npoints=None):
        
        if shape == None:
            shape = self._source_image.shape
            
        slicer = lambda : [slice(corner[i],shape[i]+corner[i],spacing[i]) for i in range(3)]
        fov = self._source_image[slicer()]

        # Adjust spacing to match desired number of points
        if fixed_npoints: 
            spacing = subsample(fov.data, npoints=fixed_npoints)
            fov = self._source_image[slicer()]

        self._source = fov.data
        self._source_npoints = (fov.data >= 0).sum()
        self._source_toworld = fov.affine

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

    """
    For non-affine transformations, we'll do:

    XYZ = voxel locations in self._source 
    apply_affine(self._target_fromworld, T(apply_affine(self._source_toworld, XYZ)))
    
    """


    def eval(self, T):
        Tv = self.voxel_transform(T)
        seed = self._interp
        if self._interp < 0:
            seed = - np.random.randint(maxint)
        _joint_histogram(self._joint_hist, 
                         self._source.flat, ## array iterator
                         self._target, 
                         Tv,
                         1, ## affine 
                         seed)
        #self.source_hist = np.sum(self._joint_hist, 1)
        #self.target_hist = np.sum(self._joint_hist, 0)
        return _similarity(self._joint_hist, 
                           self._source_hist, 
                           self._target_hist, 
                           self._similarity, 
                           self._pdf, 
                           self._similarity_func)

    ## FIXME: check that the dimension of start is consistent with the search space. 
    def optimize(self, search='affine', method='powell', start=None, 
                 radius=100, tol=1e-1, ftol=1e-2):

        """
        radius: a parameter for the 'typical size' in mm of the object
        being registered. This is used to reformat the parameter
        vector (translation+rotation+scaling+shearing) so that each
        element roughly represents a variation in mm.
        """
        if start == None: 
            T = Affine()
        else:
            T = Affine(start.vec12)
        T.parametrize(subtype=search, radius=radius)
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

        if method=='simplex':
            print ('Optimizing using the simplex method...')
            tc = fmin(loss, tc0, callback=callback, xtol=tol, ftol=ftol)
        elif method=='powell':
            print ('Optimizing using Powell method...') 
            tc = fmin_powell(loss, tc0, callback=callback, xtol=tol, ftol=ftol)
        elif method=='conjugate_gradient':
            print ('Optimizing using conjugate gradient descent...')
            tc = fmin_cg(loss, tc0, callback=callback, gtol=ftol)
        else:
            raise ValueError('Unrecognized optimizer')
        
        # Output
        T.param = tc
        return T 

    # Return a set of similarity
    def explore(self, T0, 
                ux=[0], uy=[0], uz=[0],
                rx=[0], ry=[0], rz=[0], 
                sx=[0], sy=[0], sz=[0],
                qx=[0], qy=[0], qz=[0]):
        
        grids = np.mgrid[0:len(ux), 0:len(uy), 0:len(uz), 
                         0:len(rx), 0:len(ry), 0:len(rz), 
                         0:len(sx), 0:len(sy), 0:len(sz), 
                         0:len(qx), 0:len(qy), 0:len(qz)]
        ntrials = np.prod(grids.shape[1:])
        params = [ux, uy, uz, rx, ry, rz, sx, sy, sz, qx, qy, qz]
        Params = [np.asarray(params[i])[grids[i,:]].ravel() for i in range(len(params))]
        
        simis = np.zeros(ntrials)
        vec12s = np.zeros([12, ntrials])

        T = Affine()
        for i in range(ntrials):
            t = T0.vec12 + np.array([p[i] for p in Params])
            T.vec12 = t 
            simis[i] = self.eval(T)
            vec12s[:, i] = t 

        return simis, vec12s
        

def clamp(x, bins=256):
    """ 
    Clamp array values that fall within a given mask in the range
    [0..bins-1] and reset masked values to -1.
    
    Parameters
    ----------
    x : ndarray
      The input array

    bins : number 
      Desired number of bins
    
    Returns
    -------
    y : ndarray
      Clamped array

    bins : number 
      Adjusted number of bins 

    """
    
    # Create output array to allow in-place operations
    y = np.zeros(x.shape, dtype=_clamp_dtype)

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


