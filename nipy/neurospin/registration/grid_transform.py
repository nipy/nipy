from nipy.neurospin.image import apply_affine

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
        if affine == None: 
            self._affine = toworld 
        else: 
            self._affine = np.dot(affine, toworld)
        self._set_param(np.zeros(len(data)))
        
    def _get_data(self): 
        return self._data

    def _get_affine(self): 
        return self._affine

    def _get_param(self):
        return self._param

    def _set_param(self, p):
        # Specify dtype to allow in-place operations
        self._param = np.asarray(p, dtype='double') 

    def __getitem__(self, slices):
        """
        Return the sampled displacements on the subgrid specified by
        slices. 
        """
        tmp = self._param[0]*self.data[0][slices]
        for i in np.arange(1, self._param.size):
            tmp += self._param[i]*self.data[i][slices]
        XYZ = np.c_[[c.ravel() for c in np.mgrid[slices]]].T # Nx3 array
        tmp += apply_affine(self._affine, XYZ).reshape(tmp.shape)

        return tmp

    data = property(_get_data)
    toworld = property(_get_affine)
    param = property(_get_param, _set_param) 
    




"""
data = [np.random.rand(20,20,10,3) for i in range(5)]
g = GridTransform(data, np.eye(4))
"""
