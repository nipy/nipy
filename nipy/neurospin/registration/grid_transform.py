from nipy.neurospin.image import Image, apply_affine, subgrid_affine, inverse_affine

import numpy as np 
from scipy.ndimage import gaussian_filter


class GridTransform(object): 

    def __init__(self, image, data, affine=None): 
        """
        image : a neurospin image or a sequence (shape, affine)
        
        data : a 5d array representing the deformation modes, first
        dimension should represent the mode index, next three
        dimensions should represent space, last dimension should be 3.
        """
        self._generic_init(image, affine)
        self._data = data 
        self._set_param(np.zeros(len(data)))

    def _generic_init(self, image, affine): 
        if isinstance(image, Image): 
            self._shape = image.shape
            self._toworld = image.affine
        else:
            self._shape, self._toworld = image
        if affine == None:
            self._affine = np.eye(4)
            self._grid_affine = self._toworld
        else:
            self._affine = affine
            self._grid_affine = np.dot(affine, self._toworld)
        self._IJK = None
        self._sampled = None

    def _get_shape(self): 
        return self._shape

    def _get_affine(self):
        return self._affine

    def _get_param(self):
        return self._param

    def _set_param(self, p):
        # Specify dtype to allow in-place operations
        self._param = np.asarray(p, dtype='double') 

    def __getitem__(self, slices):
        data = self._data[[slice(0,None)]+list(slices)]
        toworld = subgrid_affine(self._toworld, slices)
        return GridTransform((data.shape[1:-1], toworld), data, self._affine)

    def IJK(self):
        if not self._IJK == None:
            return self._IJK 
        tmp = np.mgrid[[slice(0, s) for s in self._shape]]
        self._IJK = np.rollaxis(tmp, 0, 1+len(self._shape))
        return self._IJK 
        
    def sample_affine(self): 
        if self._sampled == None: 
            self._sampled = apply_affine(self._grid_affine, self.IJK())
        else: 
            self._sampled[:] = apply_affine(self._grid_affine, self.IJK())

    def __call__(self):
        """
        Return the displacements sampled on the grid. 
        """
        self.sample_affine()
        self._sampled += np.sum((self._data.T*self._param).T, 0)
        return self._sampled

    shape = property(_get_shape)
    affine = property(_get_affine)
    param = property(_get_param, _set_param)
    


"""
A derivation of the GridTransform class for which data is not pre-computed
"""

class SplineTransform(GridTransform):

    def __init__(self, image, control_points, sigma, grid_coords=False, affine=None):
        """
        control_points: a Nx3 array of world coordinates

        if grid_coords is True, both `control_points` and `sigma` are
        interpreted in voxel coordinates.
        """
        self._generic_init(image, affine)
        fromworld = inverse_affine(self._toworld)
        if grid_coords:
            self._control_points = apply_affine(self._toworld, control_points)
            tmp = control_points
            self._sigma = np.abs(np.diagonal(self._toworld)[0:-1]*sigma)
            self._grid_sigma = sigma*np.ones(3)
        else:
            self._control_points = np.asarray(control_points)
            tmp = apply_affine(fromworld, control_points)
            self._sigma = sigma*np.ones(3) 
            self._grid_sigma = np.abs(np.diagonal(fromworld)[0:-1]*sigma)

        # TODO : make sure the control point indices fall within the
        # subgrid and maybe raise a warning if rounding is too severe

        tmp = np.round(tmp).astype('int')
        self._idx_control_points = tuple([tmp[:,:,:,i] for i in range(tmp.shape[3])])
        
        self._norma = np.sqrt(2*np.pi)*self._grid_sigma
        self._set_param(np.zeros(3*self._control_points.shape[0]))


        """
        if basis in builtin_bases:
            self._basis = builtin_bases[basis]
        else:
            self._basis = basis
        self._kwargs = kwargs
        """

    def __getitem__(self, slices):
        toworld = subgrid_affine(self._toworld, slices)
        fake_data = np.ones(self._shape, dtype='bool')[slices]
        return SplineTransform((fake_data.shape, toworld), 
                               self._control_points, self._sigma,
                               grid_coords=False, affine=self._affine)

    def __call__(self): 
        
        self.sample_affine()
        I = np.zeros(self._shape)
        
        for i in range(3): 
            I[self._idx_control_points] = self._param[i::3]
            self._sampled[:,:,:,i] += self._norma[i]*gaussian_filter(I, sigma=self._grid_sigma[i])
        return self._sampled





"""
builtin_bases = {'gaussian': gaussian}
"""
