"""
Intensity-based matching. 

Questions: alexis.roche@gmail.com
"""
from nipy.neurospin.image import Image
from registration_module import _joint_histogram, _similarity, similarity_measures
from affine import Affine, BRAIN_RADIUS_MM

import numpy as np  
import scipy as sp 
from sys import maxint

# Globals
_clamp_dtype = 'short'


"""
Input is an object that has at least an 'array'
field and either a 'transform' or a 'voxsize' field. 

array: numpy array (ndim should be at most 3)
mask: numpy array (binary)
nbins: int, nb of histogram bins
transform: (4,4) numpy array, voxel to real spatial transformation

TODO: type checking. implement proper masking.

For the time being, any transformation is assumed to be a 4x4 matrix. 
"""

# Dictionary of interpolation methods
# pv: Partial volume 
# tri: Trilinear 
# rand: Random interpolation
interp_methods = {'pv': 0, 'tri': 1, 'rand': -1}


class IconicMatcher(object):

    def __init__(self, source, target, bins = 256):

        """
        IconicMatcher class for intensity-based image registration. 
        """
        ## FIXME: test that input images are 3d

        # Binning size  
        if isinstance (bins, (int, long, float)): 
            bins = [int(bins), int(bins)]

        # Source image binning
        values, s_bins = clamp(source(), bins=bins[0])
        self.source = source.fill(values)
        self.source_data = self.source.data
 
        # Target image padding + binning
        data, t_bins = clamp(target(), bins=bins[1])
        tmp = target.fill(data)
        self.target_data = -np.ones(np.array(target.shape)+2, dtype=_clamp_dtype)
        self.target_data[1:target.shape[0]+1:, 1:target.shape[1]+1:, 1:target.shape[2]+1:] = tmp.data
        # Histograms
        self.joint_hist = np.zeros([s_bins, t_bins])
        self.source_hist = np.zeros(s_bins)
        self.target_hist = np.zeros(t_bins)

        # Image-to-world transforms 
        self.source_toworld = source.affine
        self.target_fromworld = target.inv_affine
        
        # Set default registration parameters
        self.set_interpolation()
        self.set_field_of_view()
        self.set_similarity()

    def set_interpolation(self, method='pv'):
        self.interp = method
        self._interp = interp_methods[method]

    def set_field_of_view(self, spacing=[1,1,1], corner=[0,0,0], shape=None, 
                          fixed_npoints=None):
        
        if shape == None:
            shape = self.source.shape
            
        slicer = lambda : [slice(corner[i],shape[i]+corner[i],spacing[i]) for i in range(3)]
        fov = self.source[slicer()]

        # Adjust spacing to match desired number of points
        if not fixed_npoints: 
            spacing = subsample(fov.data, npoints=fixed_npoints)
        fov = self.source[slicer()]

        self.source_fov = fov
        self.source_fov_npoints = (fov.data >= 0).sum()

        
    def set_similarity(self, similarity='cc', normalize=None, pdf=None): 
        self.similarity = similarity
        if similarity in similarity_measures: 
            self._similarity = similarity_measures[similarity]
        else: 
            self._similarity = similarity_measures['custom']
        self.normalize = normalize
        ## Use array rather than asarray to ensure contiguity 
        self.pdf = np.array(pdf)  

    def voxel_transform(self, T):
        """ 
        T is the 4x4 transformation between the real coordinate systems
        The corresponding voxel transformation is: Tv = Tt^-1 * T * Ts
        """
        ## C-contiguity required
        return np.dot(self.target.inverse_affine, np.dot(T, self.source.affine)) 

    def fov_voxel_transform(self, T): 
        ## C-contiguity ensured 
        return np.dot(self.target.inverse_affine, np.dot(T, self.source_fov.affine)) 

    def eval(self, T):
        Tv = self.fov_voxel_transform(T)
        seed = self._interp
        if self._interp < 0:
            seed = - np.random.randint(maxint)
        _joint_histogram(self.joint_hist, 
                         self.source_fov.data.flat, ## array iterator
                         self.target, 
                         Tv, 
                         seed)
        #self.source_hist = np.sum(self.joint_histo, 1)
        #self.target_hist = np.sum(self.joint_histo, 0)
        return _similarity(self.joint_hist, 
                           self.source_hist, 
                           self.target_hist, 
                           self._similarity, 
                           self.pdf)

    ## FIXME: check that the dimension of start is consistent with the search space. 
    def optimize(self, search='rigid', method='powell', start=None, 
                 radius=BRAIN_RADIUS_MM, tol=1e-1, ftol=1e-2):
        """
        radius: a parameter for the 'typical size' in mm of the object
        being registered. This is used to reformat the parameter
        vector (translation+rotation+scaling+shearing) so that each
        element roughly represents a variation in mm.
        """
        if start == None: 
            T = Affine(subtype=search, radius=radius)
        else:
            T = Affine(subtype=search, vec12=start.vec12, radius=radius)
        tc0 = T.to_param()

        # Loss function to minimize
        def loss(tc):
            T.from_param(tc)
            return -self.eval(T) 
    
        def callback(tc):
            T.from_param(tc)
            print(T)
            print(self.similarity + ' = %s' % self.eval(T))
            print('')
                  

        # Switching to the appropriate optimizer
        print('Initial guess...')
        print(T)

        if method=='simplex':
            print ('Optimizing using the simplex method...')
            tc = sp.optimize.fmin(loss, tc0, callback=callback, xtol=tol, ftol=ftol)
        elif method=='powell':
            print ('Optimizing using Powell method...') 
            tc = sp.optimize.fmin_powell(loss, tc0, callback=callback, xtol=tol, ftol=ftol)
        elif method=='conjugate_gradient':
            print ('Optimizing using conjugate gradient descent...')
            tc = sp.optimize.fmin_cg(loss, tc0, callback=callback, gtol=ftol)
        else:
            raise ValueError('Unrecognized optimizer')
        
        # Output
        T.from_param(tc)
        return T 

    # Return a set of similarity
    def explore(self, 
                ux=[0], uy=[0], uz=[0],
                rx=[0], ry=[0], rz=[0], 
                sx=[1], sy=[1], sz=[1],
                qx=[0], qy=[0], qz=[0]):

        grids = np.mgrid[0:len(ux), 0:len(uy), 0:len(uz), 
                         0:len(rx), 0:len(ry), 0:len(rz), 
                         0:len(sx), 0:len(sy), 0:len(sz), 
                         0:len(qx), 0:len(qy), 0:len(qz)]

        ntrials = np.prod(grids.shape[1:])
        UX = np.asarray(ux)[grids[0,:]].ravel()
        UY = np.asarray(uy)[grids[1,:]].ravel()
        UZ = np.asarray(uz)[grids[2,:]].ravel()
        RX = np.asarray(rx)[grids[3,:]].ravel()
        RY = np.asarray(ry)[gprids[4,:]].ravel()
        RZ = np.asarray(rz)[grids[5,:]].ravel()
        SX = np.asarray(sx)[grids[6,:]].ravel()
        SY = np.asarray(sy)[grids[7,:]].ravel()
        SZ = np.asarray(sz)[grids[8,:]].ravel()
        QX = np.asarray(qx)[grids[9,:]].ravel()
        QY = np.asarray(qy)[grids[10,:]].ravel()
        QZ = np.asarray(qz)[grids[11,:]].ravel()
        simis = np.zeros(ntrials)
        vec12s = np.zeros([12, ntrials])

        T = Affine()
        for i in range(ntrials):
            t = np.array([UX[i], UY[i], UZ[i],
                          RX[i], RY[i], RZ[i],
                          SX[i], SY[i], SZ[i],
                          QX[i], QY[i], QZ[i]])
            T.set_vec12(t)
            simis[i] = self.eval(T)
            vec12s[:, i] = t 

        return simis, vec12s
        

def clamp(y, im, bins=256):
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
        y[:] = np.round(a*(values-th))
 
    return y, bins 


def subsample(im, npoints):
    """  
    Tune spacing factors so that the number of voxels in the output
    block matches a given number.
    
    Parameters
    ----------
    im : ndarray or sequence  
      Image to subsample
    
    npoints : number
      Target number of voxels (negative values will be ignored)

    Returns
    -------
    spacing: ndarray 
      Spacing factors
                 
    """
    dims = im.shape
    actual_npoints = (im >= 0).sum()
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
            
    return spacing
