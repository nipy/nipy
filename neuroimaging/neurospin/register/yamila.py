"""
YAMILA = Yet Another Mutual Information-Like Aligner

Questions: alexis.roche@gmail.com
"""
import _yamila
import affine_transform

import numpy as np  
import scipy as sp 
import scipy.optimize 
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


# Dictionary of interpolation methods
interp_methods = {'partial volume': 0,
                  'trilinear': 1,
                  'random': -1}



def clamp(x, th=0, mask=None, bins=256): 
    """
    Define a mask as the intersection of the input mask and
    (x>=th). Then, clamp in-mask array values in the range [0..bins-1]
    and reset out-of-mask values to -1.
    """
    # Mask 
    if mask == None: 
        mask = np.ones(x.shape, dtype='bool')
    mask *= (x>=th)

    # Create output array to allow in-place operations
    y = np.zeros(x.shape, dtype='short')
    dmaxmax = 2**(8*y.dtype.itemsize-1)-1

    # Threshold
    dmax = bins-1 ## default output maximum value
    if dmax > dmaxmax: 
        raise ValueError("Excessive number of bins.")
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
    xmax is translated to xmax-tth casted to dtype='short'. Otherwise,
    compress after downshifting image values (values equal to the
    threshold are reset to zero). 
    """
    y[mask==False] = -1
    if np.issubdtype(x.dtype, int) and d<=dmax:
        y[mask] = x[mask]-th
        bins = int(d)+1
    else: 
        a = dmax/d
        y[mask] = np.round(a*(x[mask]-th))
 
    return y, bins 



def _similarity(similarity, normalize): 

    if normalize == None:
        k = 0
    elif normalize == 'sophia':
        k = 1
    elif normalize == 'saclay':
        k = 2
    flag = _yamila.similarity_measures[similarity][k] ## FIXME: may not exist, should check! 
    return flag



def _interp(interp):

    flag = interp_methods[interp]
    return flag




class JointHistogram():

    def __init__(self, source, target, 
                 source_transform, target_transform,
                 source_threshold=0, target_threshold=0,  
                 source_mask=None, target_mask=None,
                 bins=256):

        """
        JointHistogram class for intensity-based image registration. 
        """
        ## FIXME: test that input images are 3d

        # Source image binning
        self.source = source
        self.source_clamped, s_bins = clamp(source, th=source_threshold, mask=source_mask, bins)

        # Target image padding + binning
        self.target = target 
        self.target_clamped = -np.ones(np.array(target.shape)+2)
        self.target_clamped[1:target.shape[0]-1:, 1:target.shape[1]-1:, 1:target.shape[2]-1:], \
            t_bins = clamp(target, th=target_threshold, mask=target_mask, bins)
        
        # Histograms
        self.joint_hist = np.zeros([s_bins, t_bins])
        self.source_hist = np.zeros(s_bins)
        self.target_hist = np.zeros(t_bins)
        
        # Image-to-world transforms 
        self.source_transform = source_transform
        self.target_transform_inv = np.linalg.inv(target_transform)

        # Set default registration parameters
        self.set()

    # Use array rather than asarray to ensure contiguity 
    def set(self, interp='partial volume', 
            similarity='correlation coefficient', normalize=None, 
            subsampling=[1,1,1], corner=[0,0,0], size=None, pdf=None):
        self.block_subsampling = np.array(subsampling, dtype='uint')
        self.block_corner = np.array(corner, dtype='uint')
        if size == None:
            size = self.source.shape
        self.block_size = np.array(size, dtype='uint')
        self.source_block = self.source_clamped[corner[0]:corner[0]+size[0]-1:subsampling[0],
                                                corner[1]:corner[1]+size[1]-1:subsampling[1],
                                                corner[2]:corner[2]+size[2]-1:subsampling[2]]
        self.block_npoints = (self.source_block >= 0).sum()

        ## Taux: block to full array transformation
        Taux = np.diag(np.concatenate((self.block_subsampling,[1]),1))
        Taux[0:3,3] = self.block_corner
        self.block_transform = np.dot(self.source_transform, Taux)
        self.interp = interp
        self._interp = _interp(interp)
        self.similarity = similarity
        self.normalize = normalize
        self._similarity = _similarity(similarity, normalize)
        self.pdf = np.array(pdf)        

    # T is the 4x4 transformation between the real coordinate systems
    # The corresponding voxel transformation is: Tv = Tt^-1 o T o Ts
    def voxel_transform(self, T):
        return np.dot(self.target_transform_inv, 
                      np.dot(T, self.source_transform)) ## C-contiguity ensured

    def block_voxel_transform(self, T): 
        return np.dot(self.target_transform_inv, 
                      np.dot(T, self.block_transform)) ## C-contiguity ensured 

    def eval(self, T):
        Tv = self.block_voxel_transform(T)
        seed = self._interp
        if self._interp < 0:
            seed = - np.random.randint(maxint)
        _yamila.joint_hist(self.joint_hist, 
                           self.source_block.flat, ## array iterator
                           self.target_clamped, 
                           Tv, 
                           seed)
        #self.source_hist = np.sum(self.joint_histo, 1)
        #self.target_hist = np.sum(self.joint_histo, 0)
        return _yamila.similarity(self.joint_hist, self.source_hist, 
                                  self.target_hist, self._similarity, 
                                  self.block_npoints, self.pdf)

    ## FIXME: check that the dimension of start is consistent with the search space. 
    def optimize(self, search='rigid 3D', method='powell', start=None, radius=10):
        """
        radius: a parameter for the 'typical size' in mm of the object
        being registered. This is used to reformat the parameter vector
        (translation+rotation+scaling+shearing) so that each element
        represents a variation in mm.
        """
        
        # Constants
        precond = affine_transform.preconditioner(radius)
        if start == None: 
            t0 = np.array([0, 0, 0, 0, 0, 0, 1., 1., 1., 0, 0, 0])
        else:
            t0 = np.asarray(start)

        # Search space
        stamp = affine_transform.transformation_types[search]
        tc0 = affine_transform.vector12_to_param(t0, precond, stamp)

        # Loss function to minimize
        def loss(tc):
            t = affine_transform.param_to_vector12(tc, t0, precond, stamp)
            return(-self.eval(affine_transform.matrix44(t)))
    
        def print_vector12(tc):
            t = affine_transform.param_to_vector12(tc, t0, precond, stamp)
            print('')
            print ('  translation : %s' % t[0:3].__str__())
            print ('  rotation    : %s' % t[3:6].__str__())
            print ('  scaling     : %s' % t[6:9].__str__())
            print ('  shearing    : %s' % t[9:12].__str__())
            print('')

        # Switching to the appropriate optimizer
        print('Initial guess...')
        print_vector12(tc0)

        if method=='simplex':
            print ('Optimizing using the simplex method...')
            tc = sp.optimize.fmin(loss, tc0, callback=print_vector12)
        elif method=='powell':
            print ('Optimizing using Powell method...') 
            tc = sp.optimize.fmin_powell(loss, tc0, callback=print_vector12)
        elif method=='conjugate gradient':
            print ('Optimizing using conjugate gradient descent...')
            tc = sp.optimize.fmin_cg(loss, tc0, callback=print_vector12)
        else:
            raise ValueError, 'Unrecognized optimizer'
        
        # Output
        t = affine_transform.param_to_vector12(tc, t0, precond, stamp)
        T = affine_transform.matrix44(t)
        return (T, t)

    # Return a set of similarity
    def explore(self, ux=[0], uy=[0], uz=[0], rx=[0], ry=[0], rz=[0], 
                sx=[1], sy=[1], sz=[1], qx=[0], qy=[0], qz=[0]):

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
        similarities = np.zeros(ntrials)
        params = np.zeros([12, ntrials])

        for i in range(ntrials):
            t = np.array([UX[i], UY[i], UZ[i],
                          RX[i], RY[i], RZ[i],
                          SX[i], SY[i], SZ[i],
                          QX[i], QY[i], QZ[i]])

            similarities[i] = self.eval(affine_transform.matrix44(t))
            params[:, i] = t 

        return similarities, params
        

    def resample(self, T, toresample='source', dtype=None):
        if toresample is 'target': 
            Tv = self.voxel_transform(T)
            out = affine_transform.resample(self.target, self.source.shape, Tv, 
                                  datatype=dtype)
        else:
            Tv_inv = np.linalg.inv(self.voxel_transform(T))
            out = affine_transform.resample(self.source, self.target.shape, Tv_inv, 
                                  datatype=dtype)

        return out


