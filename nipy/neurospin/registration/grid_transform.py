from nipy.neurospin.image import apply_affine, subgrid_affine

import numpy as np 


def gauss(XYZ, c, s):
    tmp = (XYZ[0]-c[0])**2
    for i in np.arange(1, len(XYZ)): 
        tmp += (XYZ[i]-c[i])**2
    return np.exp(-.5*tmp/s**2)
    

class GridTransform(object): 

    def __init__(self, data, toworld, affine=None): 
        """
        data : a sequence of 4d arrays representing the deformation
        modes, last dimensions should be 3. 

        toworld : 4x4 array describing the grid-to-world affine
        transformation.
        """
        self._data = data 
        self._shape = data[0].shape
        self._toworld = toworld
        if affine == None: 
            affine = np.eye(4)
        self._affine = affine
        self._grid_affine = np.dot(affine, toworld)
        self._set_param(np.zeros(len(data)))
        
    def _get_data(self): 
        return self._data

    def _get_shape(self): 
        return self._shape

    def _get_affine(self): 
        return self._affine

    def _get_param(self):
        return self._param

    def _set_param(self, p):
        # Specify dtype to allow in-place operations
        self._param = np.asarray(p, dtype='double') 

    def getitem__(self, slices):
        data = [self._data[i] for i in range(len(data))]
        toworld = subgrid_affine(self._toworld, slices)
        return GridTransform(data, toworld, self._affine)

    def __call__(self):
        """
        Return the displacements sampled on the grid. 
        """
        tmp = self._param[0]*self._data[0]
        for i in np.arange(1, self._param.size):
            tmp += self._param[i]*self.data[i]
        # Add the affine component...
        slices = [slice(0, s) for s in self._shape]
        XYZ = np.c_[[c.ravel() for c in np.mgrid[slices]]].T # Nx3 array
        tmp += apply_affine(self._grid_affine, XYZ).reshape(tmp.shape)
        return tmp

    data = property(_get_data)
    shape = property(_get_shape)
    affine = property(_get_affine)
    param = property(_get_param, _set_param) 
    




"""
data = [np.random.rand(20,20,10,3) for i in range(5)]
g = GridTransform(data, np.eye(4))
"""
