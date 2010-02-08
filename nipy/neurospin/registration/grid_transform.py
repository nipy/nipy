from nipy.neurospin.image import apply_affine

import numpy as np 


def gauss(XYZ, c, s):
    tmp = (XYZ[0]-c[0])**2
    for i in np.arange(1, len(XYZ)): 
        tmp += (XYZ[i]-c[i])**2
    return np.exp(-.5*tmp/s**2)
    

class GridTransform(object): 


    def __init__(self, modes, toworld, subgrid=None):
        """
        modes : a sequence of 4d-arrays with same shape and last
        dimension 1 or 3

        toworld : 4x4 matrix

        subgrid : tuple of 3d indices 
        """
        shape = modes[0].shape
        ## TODO: check all same shape...
        if subgrid == None:
            subgrid = tuple(np.mgrid[[slice(0,d) for d in shape]])
        dim = len(subgrid)
        subgrid = [subgrid[i].ravel() for i in range(dim)]
        self.modes = tuple([m[subgrid].T for m in modes])
        self.base = np.asarray(apply_affine(toworld, subgrid))
        self._set_param(np.zeros(len(modes)))

    def __array__(self, dtype='double'): 
        tmp = self.base.copy()
        for i in np.arange(0, self.param.size):
            tmp += self.param[i]*self.modes[i]
        return tmp.astype(dtype)
        
    def _get_param(self):
        return self._param

    def _set_param(self, p):
        # Specify dtype to allow in-place operations
        self._param = np.asarray(p, dtype='double') 

    param = property(_get_param, _set_param) 

    def __call__(self, XYZ):
        return 


"""
shape = (5,5,5)
mask = tuple(np.mgrid[[slice(0,d) for d in shape]]) 

modes = [np.random.rand(20,20,10,3) for i in range(5)]

g = GridTransform(modes, np.eye(4), mask)
"""
