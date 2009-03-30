import _iconic
import transform_affine as affine

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


def _format(im, mask=0, nbins=256):

    array = im.array
    transform = im.transform

    # Convert mask to intensity threshold
    if isinstance(mask, int) or isinstance(mask, long) or isinstance(mask, float):
        thres = float(mask)
    else:
        thres = 0.0 ## FIXME!!

    return array, thres, nbins, transform


def _similarity(similarity, normalize): 

    if normalize == None:
        k = 0
    elif normalize == 'sophia':
        k = 1
    elif normalize == 'saclay':
        k = 2
    flag = _iconic.similarity_measures[similarity][k] ## FIXME: may not exist, should check! 
    return flag



def _interp(interp):

    flag = interp_methods[interp]
    return flag


class iconic():

    def __init__(self, source, target):
        self.source, s_thres, s_nbins, s_transform = _format(source)
        self.target, t_thres, t_nbins, t_transform = _format(target)
        self.source_clamped, self.target_clamped, \
            self.joint_hist, self.source_hist, self.target_hist = \
            _iconic.imatch(self.source, self.target, s_thres, t_thres, 
                            s_nbins, t_nbins)
        self.source_transform = s_transform
        self.target_transform_inv = np.linalg.inv(t_transform)
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
        ## Taux: block to full array transformation
        Taux = np.diag(np.concatenate((self.block_subsampling,[1]),1))
        Taux[0:3,3] = self.block_corner
        self.block_transform = np.dot(self.source_transform, Taux)
        self.interp = interp
        self._interp = _interp(interp)
        self.similarity = similarity
        self.normalize = normalize
        self._similarity = _similarity(similarity, normalize)
        self.block_npoints = _iconic.block_npoints(self.source_clamped, 
                    self.block_subsampling, self.block_corner, self.block_size)
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
        Tb = self.block_voxel_transform(T)
        seed = self._interp
        if self._interp < 0:
            seed = - np.random.randint(maxint)
        _iconic.joint_hist(self.joint_hist, self.source_clamped, 
                           self.target_clamped, Tb, self.block_subsampling, 
                           self.block_corner, self.block_size, seed)
        #self.source_hist = np.sum(self.joint_histo, 1)
        #self.target_hist = np.sum(self.joint_histo, 0)
        return _iconic.similarity(self.joint_hist, self.source_hist, 
                                  self.target_hist, self._similarity, 
                                  self.block_npoints, self.pdf)

    ## TODO : check that the dimension of start is consistent with the search space. 
    def optimize(self, search='rigid 3D', method='powell', start=None, radius=10):
        """
        radius: a parameter for the 'typical size' in mm of the object
        being registered. This is used to reformat the parameter vector
        (translation+rotation+scaling+shearing) so that each element
        represents a variation in mm.
        """
        
        # Constants
        precond = affine.preconditioner(radius)
        if start == None: 
            t0 = np.array([0, 0, 0, 0, 0, 0, 1., 1., 1., 0, 0, 0])
        else:
            t0 = np.asarray(start)

        # Search space
        stamp = affine.transformation_types[search]
        tc0 = affine.vector12_to_param(t0, precond, stamp)

        # Loss function to minimize
        def loss(tc):
            t = affine.param_to_vector12(tc, t0, precond, stamp)
            return(-self.eval(affine.matrix44(t)))
    
        def print_vector12(tc):
            t = affine.param_to_vector12(tc, t0, precond, stamp)
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
        t = affine.param_to_vector12(tc, t0, precond, stamp)
        T = affine.matrix44(t)
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

            similarities[i] = self.eval(affine.matrix44(t))
            params[:, i] = t 

        return similarities, params
        

    def resample(self, T, toresample='source', dtype=None):
        if toresample is 'target': 
            Tv = self.voxel_transform(T)
            out = affine.resample(self.target, self.source.shape, Tv, 
                                  datatype=dtype)
        else:
            Tv_inv = np.linalg.inv(self.voxel_transform(T))
            out = affine.resample(self.source, self.target.shape, Tv_inv, 
                                  datatype=dtype)

        return out


