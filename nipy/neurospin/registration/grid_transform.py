from nipy.neurospin.image import Image, apply_affine, subgrid_affine

import numpy as np 
    

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
        self._sample_affine = None

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
        if not self._sample_affine == None:
            return self._sample_affine
        self._sample_affine = apply_affine(self._grid_affine, self.IJK())
        return self._sample_affine
        
    def __call__(self):
        """
        Return the displacements sampled on the grid. 
        """
        tmp = self.sample_affine().copy()
        tmp += np.sum((self._data.T*self._param).T, 0)
        return tmp 

    shape = property(_get_shape)
    affine = property(_get_affine)
    param = property(_get_param, _set_param)
    


"""
A derivation of the GridTransform class for which data is not pre-computed
"""

class SplineTransform(GridTransform):

    def __init__(self, image, control_points, affine=None, basis='gaussian', **kwargs):
        """
        control_points: a Nx3 array of world coordinates 
        """
        self._generic_init(image, affine)
        self._control_points = np.asarray(control_points)
        if basis in builtin_bases:
            self._basis = builtin_bases[basis]
        else:
            self._basis = basis
        self._kwargs = kwargs

        """
        - Precompute grid coordinates of control points
        - Precompute sigma in grid coordinates
        - Afterwards:

        I = np.zeros(self._shape)
        res = np.zeros(list(self._shape)+[3])

        for i in range(3)...
        
          p = self._param[i::3]
          I[grid_control_points] = p

          res[:,:,:,i] = nd.gaussian_filter(I)

        """

    def mode(self, i):
        return self._basis(self.XYZ(), self._control_points[i], **self._kwargs)
            

    def __getitem__(self, slices):
        toworld = subgrid_affine(self._toworld, slices)
        return SplineTransform((data.shape[1:-1], toworld), data, self._affine)



def gaussian(XYZ, c, scale=1.):
    tmp = (XYZ-c)/scale
    return np.exp(-.5*tmp**2)


builtin_bases = {'gaussian': gaussian}


"""
data = [np.random.rand(20,20,10,3) for i in range(5)]
g = GridTransform(data, np.eye(4))
"""
