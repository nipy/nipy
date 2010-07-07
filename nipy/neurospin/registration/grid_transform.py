# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from nipy.neurospin.image import Image, apply_affine, subgrid_affine, inverse_affine

import numpy as np 
from scipy.ndimage import gaussian_filter


class GridTransform(object): 

    def __init__(self, image, data, affine=None): 
        """
        image : a neurospin image or a sequence (shape, affine)
        
        data : a 5d array representing the deformation modes, first
        three dimensions should represent space, next dimension should
        be the mode index, last dimension should be 3.
        """
        nparams = data.shape[3]
        self._generic_init(image, affine, nparams)
        self._data = data 

    def _generic_init(self, image, affine, nparams): 
        if isinstance(image, Image): 
            self._shape = image.shape
            self._toworld = image.affine
        else:
            self._shape, self._toworld = image
        if affine == None:
            self._affine = np.eye(4)
            self._grid_affine = self._toworld
        else:
            self._affine = np.asarray(affine)
            self._grid_affine = np.dot(self._affine, self._toworld)
        self._IJK = None
        self._sampled = None
        
        self._param = np.zeros(nparams)
        self._free_param_idx = slice(0, nparams) 

    def _get_shape(self): 
        return self._shape

    def _get_affine(self):
        return self._affine

    def _get_param(self):
        return self._param[self._free_param_idx]

    def _set_param(self, p):
        # Specify dtype to allow in-place operations
        self._param[self._free_param_idx] = np.asarray(p) 

    def __getitem__(self, slices):
        data = self._data[slices]
        toworld = subgrid_affine(self._toworld, slices)
        res = GridTransform((data.shape[:-2], toworld), data, self._affine)
        res._param[:] = self._param[:]
        res._free_param_idx = self._free_param_idx
        return res

    def IJK(self):
        if not self._IJK == None:
            return self._IJK 
        tmp = np.mgrid[[slice(0, s) for s in self._shape]]
        self._IJK = np.rollaxis(tmp, 0, 1+len(self._shape))
        return self._IJK 
        
    def _sample_affine(self): 
        if self._sampled == None: 
            self._sampled = apply_affine(self._grid_affine, self.IJK())
        else: 
            self._sampled[:] = apply_affine(self._grid_affine, self.IJK())

    def __array__(self):
        """
        Return the displacements sampled on the grid. 
        """
        self._sample_affine()
        tmp = np.reshape(self._param, (self._param.size,1))
        self._sampled += np.sum(self._data*tmp, 3)
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
        nparams = np.prod(control_points.shape)
        self._generic_init(image, affine, nparams)
        fromworld = inverse_affine(self._toworld)
        if grid_coords:
            self._control_points = apply_affine(self._toworld, control_points)
            tmp = control_points
        else:
            self._control_points = np.asarray(control_points)
            tmp = apply_affine(fromworld, control_points)
            
        # TODO : make sure the control point indices fall within the
        # subgrid and maybe raise a warning if rounding is too severe

        tmp = np.round(tmp).astype('int')
        self._idx_control_points = tuple([tmp[:,:,:,i] for i in range(tmp.shape[3])])
        self._sigma = sigma*np.ones(3) 
        self._grid_sigma = np.abs(np.diagonal(fromworld)[0:-1]*sigma)
        self._norma = np.prod(np.sqrt(2*np.pi)*self._grid_sigma)

    def __getitem__(self, slices):
        toworld = subgrid_affine(self._toworld, slices)
        fake_data = np.ones(self._shape, dtype='bool')[slices]
        res = SplineTransform((fake_data.shape, toworld), 
                              self._control_points, self._sigma,
                              grid_coords=False, affine=self._affine)
        res._param[:] = self._param[:]
        res._free_param_idx = self._free_param_idx
        return res 

    def __array__(self): 
        
        # The trick is to note that sum_i ci G(x-xi) is equivalent to
        # the convolution of the sparse image `c` with a normalized 3d
        # Gaussian kernel. 
        self._sample_affine()
        tmp = np.zeros(self._shape)
        param = np.reshape(self._param, self._control_points.shape)
        for i in range(3): 
            tmp[self._idx_control_points] = param[:,:,:,i]
            self._sampled[:,:,:,i] += self._norma*gaussian_filter(tmp, sigma=self._grid_sigma,
                                                                  mode='constant', cval=0.0)
        return self._sampled





"""
builtin_bases = {'gaussian': gaussian}
"""
