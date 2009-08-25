"""
Intensity-based matching. 

Questions: alexis.roche@gmail.com
"""
from routines import _joint_histogram, _similarity, similarity_measures
from transform import Affine, BRAIN_RADIUS_MM

import numpy as np  
import scipy as sp 
from sys import maxint

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

CLAMP_DTYPE = 'short'

# Dictionary of interpolation methods
# pv: Partial volume 
# tri: Trilinear 
# rand: Random interpolation
interp_methods = {'pv': 0, 'tri': 1, 'rand': -1}



def clamp(x, th=0, mask=None, bins=256):
    """ 
    Define a mask as the intersection of an initial mask and those
    indices for which array values are above a given threshold. Then,
    clamp in-mask array values in the range [0..bins-1] and reset
    out-of-mask values to -1.
    
    Parameters
    ----------
    x : ndarray
        The input array
    th : number
         Low threshold
    mask : ndarray
           Mask 
    bins : number 
           Desired number of bins
    
    Returns
    -------
    y : ndarray
        Clamped array
    bins : number 
           Adjusted number of bins 

    """
 
    # Mask 
    if mask == None: 
        mask = np.ones(x.shape, dtype='bool')
    mask *= (x>=th)

    # Create output array to allow in-place operations
    y = np.zeros(x.shape, dtype=CLAMP_DTYPE)
    dmaxmax = 2**(8*y.dtype.itemsize-1)-1

    # Threshold
    dmax = bins-1 ## default output maximum value
    if dmax > dmaxmax: 
        raise ValueError('Excess number of bins')
    xmin = float(x[mask].min())
    xmax = float(x[mask].max())
    th = np.maximum(th, xmin)
    if th > xmax:
        th = xmin  
        print("Warning: Inconsistent threshold %f, ignored." % th) 
    
    # Input array dynamic
    d = xmax-th

    """
    If the image dynamic is small, no need for compression: just
    downshift image values and re-estimate the dynamic range (hence
    xmax is translated to xmax-tth casted to the appropriate
    dtype. Otherwise, compress after downshifting image values (values
    equal to the threshold are reset to zero).
    """
    y[mask==False] = -1
    if np.issubdtype(x.dtype, int) and d<=dmax:
        y[mask] = x[mask]-th
        bins = int(d)+1
    else: 
        a = dmax/d
        y[mask] = np.round(a*(x[mask]-th))
 
    return y, bins 

def controlled_subsampling(source, npoints):
    """  
    Tune subsampling factors so that the number of voxels involved in
    registration match a given number.
    
    Parameters
    ----------
    source : ndarray or sequence  
             Source image to subsample
    
    npoints : number
              Target number of voxels (negative values will be ignored)

    Returns
    -------
    subsampling: ndarray 
                 Subsampling factors
                 
    sub_source: ndarray 
                Subsampled source 

    actual_npoints: number 
                 Actual size of the subsampled array 

    """
    dims = source.shape
    actual_npoints = (source >= 0).sum()
    subsampling = np.ones(3, dtype='uint')

    while actual_npoints > npoints:
        # Subsample the direction with the highest number of samples
        ddims = dims/subsampling
        if ddims[0] >= ddims[1] and ddims[0] >= ddims[2]:
            dir = 0
        elif ddims[1] > ddims[0] and ddims[1] >= ddims[2]:
            dir = 1
        else:
            dir = 2
        subsampling[dir] += 1
        sub_source = source[::subsampling[0], ::subsampling[1], ::subsampling[2]]
        actual_npoints = (sub_source >= 0).sum()
            
    return subsampling, sub_source, actual_npoints



class IconicMatcher:

    def __init__(self, 
                 source, target, 
                 source_toworld, target_toworld,
                 source_threshold=0, target_threshold=0,  
                 source_mask=None, target_mask=None,
                 bins=256):

        """
        IconicMatcher class for intensity-based image registration. 
        """
        ## FIXME: test that input images are 3d

        # Source image binning
        self.source = source
        self.source_clamped, s_bins = clamp(source, th=source_threshold, mask=source_mask, bins=bins)

        # Target image padding + binning
        self.target = target 
        self.target_clamped = -np.ones(np.array(target.shape)+2, dtype=CLAMP_DTYPE)
        aux, t_bins = clamp(target, th=target_threshold, mask=target_mask, bins=bins)
        self.target_clamped[1:target.shape[0]+1:, 1:target.shape[1]+1:, 1:target.shape[2]+1:] = aux
        
        # Histograms
        self.joint_hist = np.zeros([s_bins, t_bins])
        self.source_hist = np.zeros(s_bins)
        self.target_hist = np.zeros(t_bins)
        
        # Image-to-world transforms 
        self.source_toworld = source_toworld
        self.target_fromworld = np.linalg.inv(target_toworld)

        # Set default registration parameters
        self.set_interpolation()
        self.set_field_of_view()
        self.set_similarity()

    def set_interpolation(self, method='pv'):
        self.interp = method
        self._interp = interp_methods[method]

    def set_field_of_view(self, subsampling=[1,1,1], corner=[0,0,0], size=None, fixed_npoints=None):
        self.block_corner = np.array(corner, dtype='uint')
        if size == None:
            size = self.source.shape
        self.block_size = np.array(size, dtype='uint')
        if isinstance(fixed_npoints, int):
            self.block_subsampling, self.source_block, self.block_npoints = \
                controlled_subsampling(self.source_clamped[corner[0]:corner[0]+size[0]-1,
                                                           corner[1]:corner[1]+size[1]-1,
                                                           corner[2]:corner[2]+size[2]-1], 
                                          npoints=fixed_npoints)
        else: 
            self.block_subsampling = np.array(subsampling, dtype='uint')
            self.source_block = self.source_clamped[corner[0]:corner[0]+size[0]-1:subsampling[0],
                                                    corner[1]:corner[1]+size[1]-1:subsampling[1],
                                                    corner[2]:corner[2]+size[2]-1:subsampling[2]]
            self.block_npoints = (self.source_block >= 0).sum()
        ## Taux: block to full array transformation
        Taux = np.diag(np.concatenate((self.block_subsampling,[1]),1))
        Taux[0:3,3] = self.block_corner
        self.block_transform = np.dot(self.source_toworld, Taux)

    def set_similarity(self, similarity='cc', normalize=None, pdf=None): 
        self.similarity = similarity
        self._similarity = similarity_measures[similarity]
        self.normalize = normalize
        ## Use array rather than asarray to ensure contiguity 
        self.pdf = np.array(pdf)  

    def voxel_transform(self, T):
        """ 
        T is the 4x4 transformation between the real coordinate systems
        The corresponding voxel transformation is: Tv = Tt^-1 * T * Ts
        """
        ## C-contiguity required
        return np.dot(self.target_fromworld, np.dot(T, self.source_toworld)) 

    def block_voxel_transform(self, T): 
        ## C-contiguity ensured 
        return np.dot(self.target_fromworld, np.dot(T, self.block_transform)) 

    def eval(self, T):
        Tv = self.block_voxel_transform(T)
        seed = self._interp
        if self._interp < 0:
            seed = - np.random.randint(maxint)
        _joint_histogram(self.joint_hist, 
                         self.source_block.flat, ## array iterator
                         self.target_clamped, 
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
    def optimize(self, search='rigid', method='powell', start=None, radius=BRAIN_RADIUS_MM):
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
            tc = sp.optimize.fmin(loss, tc0, callback=callback)
        elif method=='powell':
            print ('Optimizing using Powell method...') 
            tc = sp.optimize.fmin_powell(loss, tc0, callback=callback)
        elif method=='conjugate_gradient':
            print ('Optimizing using conjugate gradient descent...')
            tc = sp.optimize.fmin_cg(loss, tc0, callback=callback)
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
        RY = np.asarray(ry)[grids[4,:]].ravel()
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
        




