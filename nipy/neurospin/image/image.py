"""
Image class. 

Here, image means a real-valued mapping from a regular 3D lattice. It
can be masked or not.
"""

import numpy as np
from scipy import ndimage

import nipy.io.imageformats as brifti

#from interp_module import apply_affine

# class `Grid`
class Grid(object): 
    """ An object to represent a regular 3d grid, or sub-grid,
    possibly transformed.

    Attributes
    ----------
    shape: tuple of ints
        Shape of the grid, e.g., ``(256,256,124)``

    corner: tuple of ints, optional
        Grid coordinates of the sub-grid corner

    subsample: tuple of ints, optional
        Subsampling factors 

    affine: ndarray 
        4x4 affine matrix that maps grid coordinates to desired coordinates. 
    
    """
    def __init__(self, shape, corner=(0,0,0), subsample=(1,1,1), affine=None):

        if not len(shape) == 3:
            raise ValueError('The specified shape should be of length 3.')
        if not len(corner) == 3:
            raise ValueError('The specified corner should be of length 3.')
        if not len(subsample) == 3:
            raise ValueError('The specified subsampling factors should be of length 3.')
        self._shape = tuple([int(shape[d]) for d in range(3)])
        self._corner = tuple([int(corner[d]) for d in range(3)])
        self._subsample = tuple([int(subsample[d]) for d in range(3)])
        self._affine = affine

    def get_shape(self):
        return self._shape

    def get_corner(self):
        return self._corner

    def get_subsample(self):
        return self._subsample
    
    def coords(self):
        xyz = np.mgrid[self._corner[0]:self._corner[0]+self._shape[0]:self._subsample[0],
                       self._corner[1]:self._corner[1]+self._shape[1]:self._subsample[1],
                       self._corner[2]:self._corner[2]+self._shape[2]:self._subsample[2]]
        xyz = xyz.reshape(3, np.prod(xyz.shape[1::]))
        if self._affine == None: 
            return xyz
        return tuple(apply_affine(self._affine, xyz))
                   
    def slice(self, array): 
        return array[self._corner[0]:self._corner[0]+self._shape[0]:self._subsample[0],
                     self._corner[1]:self._corner[1]+self._shape[1]:self._subsample[1],
                     self._corner[2]:self._corner[2]+self._shape[2]:self._subsample[2]]

    def __str__(self):
        return 'Grid: shape %s, corner %s, subsample %s' % (self._shape, self._corner, self._subsample) 

    """
    shape = property(get_shape)
    corner = property(get_corner)
    subsample = property(get_subsample)
    """
    


# class `Image`

class Image(object):
    """ A class representing a (real-valued???) mapping from a regular
        3D lattice.  

        Bla. 

        Attributes
        -----------
        
        Bla. 

        Notes
        ------

        Bla. 
    """
    def __init__(self, data, affine, mask=None, world=None):
        """ The base image class.

            Parameters
            ----------

            data: ndarray
                n dimensional array giving the embedded data, with the first
                three dimensions being spatial.

            affine: ndarray 
                a 4x4 transformation matrix from voxel to world.

            mask: ndarray, optional 
                A 3xN list of coordinates

            world: string, optional 
                An identifier for the real-world coordinate system, e.g. 
                'scanner' or 'mni'. 
        """

        # TODO: Check that the mask array, if provided, is consistent
        # with the data array
        self._shape = data.shape
        self._masked = True
        if mask == None: 
            self._masked = False
            mask = Grid(data.shape)
            self._data = data
        elif isinstance(mask, Grid):
            self._data = mask.slice(data)
        else:
            self._data = data[mask] 
        self._mask = mask 
        self._affine = affine
        self._world = world 

    def masked(self):
        return self._masked

    def get_shape(self):
        return self._shape

    def get_affine(self):
        return self._affine

    def get_data(self): 
        """
        Get the whole array data. 
        """
        if self._masked:
            array = np.zeros(self._shape, dtype=self._data.dtype)
            mask = self._mask
            if isinstance(mask, Grid):
                array[mask._corner[0]:mask._corner[0]+mask._shape[0]:mask._subsample[0],
                      mask._corner[1]:mask._corner[1]+mask._shape[1]:mask._subsample[1],
                      mask._corner[2]:mask._corner[2]+mask._shape[2]:mask._subsample[2]] = self._data
            else:
                array[mask] = self._data
            return array
        else:
            return self._data

    def get(self, coords, world_coords=False, interp_order=1, interp_opt=True):
        """ Return interpolated values at the points specified by coords. 

            Parameters
            ----------
            coords: ndarray or Grid
                List of coordinates. 
        
            Returns
            --------
            x: ndarray
                One-dimensional array. 

        """
        if isinstance(coords, Grid):
            if world_coords or coords.affine: 
                xyz = coords.coords()
            else: # no interpolation needed
                coords.slice()
            return 1
        else: # coords is an explicit list of points 
            return 2
        
        

    def as_volume_img(self, affine=None, shape=None, 
                                        interpolation=None):
        data = self.get_data()
        if shape is None:
            shape = data.shape[:3]
        shape = list(shape)
        if not len(shape) == 3:
            raise ValueError('The shape specified should be the shape '
                             'the 3D grid, and thus of length 3. %s was specified'
                             % shape )
        interpolation_order = self._get_interpolation_order(interpolation)
        # XXX: Need to use code to linearise the transform.
        affine = np.eye(4)
        x, y, z = np.indices(shape)
        x, y, z = apply_affine(x, y, z, affine)
        values = self.values_in_world(x, y, z)
        # We import late to avoid circular import
        from .volume_img import VolumeImg
        return VolumeImg(values, affine, 
                           self.world_space, metadata=self.metadata,
                           interpolation=self.interpolation)




    def get_world_coords(self):
        """ Return the data points coordinates in the world
            space.

            Returns
            --------
            x: ndarray
                x coordinates of the data points in world space
            y: ndarray
                y coordinates of the data points in world space
            z: ndarray
                z coordinates of the data points in world space

        """
        x, y, z = np.indices(self._data.shape[:3])
        return self.get_transform().mapping(x, y, z)


    # XXX: The docstring should be inherited


    def like_from_data(self, data):
        return self.__class__(data          = data,
                              transform     = copy.copy(self._transform),
                              metadata      = copy.copy(self.metadata),
                              interpolation = self.interpolation,
                              )


    def get_transform(self):
        """ Returns the transform object associated with the image which is a 
            general description of the mapping from the voxel space to the 
            world space.
            
            Returns
            -------
            transform : nipy.core.Transform object
        """
        return self._transform




    def values_in_world(self, x, y, z, interpolation=None):
        """ Return the values of the data at the world-space positions given by 
            x, y, z

            Parameters
            ----------
            x : number or ndarray
                x positions in world space, in other words milimeters
            y : number or ndarray
                y positions in world space, in other words milimeters.
                The shape of y should match the shape of x
            z : number or ndarray
                z positions in world space, in other words milimeters.
                The shape of z should match the shape of x
            interpolation : None, 'continuous' or 'nearest', optional
                Interpolation type used when calculating values in
                different word spaces. If None, the object's interpolation
                logic is used.

            Returns
            -------
            values : number or ndarray
                Data values interpolated at the given world position.
                This is a number or an ndarray, depending on the shape of
                the input coordinate.
        """
        interpolation_order = self._get_interpolation_order(interpolation)
        transform = self.get_transform()
        if transform.inverse_mapping is None:
            raise ValueError(
                "Cannot calculate the world values for volume data: mapping to "
                "word is not invertible."
                )
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)
        shape = list(x.shape)
        if not ((x.shape == y.shape) and (x.shape == z.shape)):
            raise ValueError('x, y and z shapes should be equal')
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()
        i, j, k = transform.inverse_mapping(x, y, z)
        data = self.get_data()
        data_shape = list(data.shape)
        n_dims = len(data_shape)
        if n_dims > 3:
            # Iter in a set of 3D volumes, as the interpolation problem is 
            # separable in the extra dimensions. This reduces the
            # computational cost
            data = np.reshape(data, data_shape[:3] + [-1])
            data = np.rollaxis(data, 3)
            values = [ ndimage.map_coordinates(slice, np.c_[i, j, k].T,
                                                  order=interpolation_order)
                       for slice in data]
            values = np.array(values)
            values = np.swapaxes(values, 0, -1)
            values = np.reshape(values, shape + data_shape[3:])
        else:
            values = ndimage.map_coordinates(data, np.c_[i, j, k].T,
                                        order=interpolation_order)
            values = np.reshape(values, shape)
        return values


def apply_affine(affine, coords):
    """
    Parameters
    ----------
    affine: ndarray 
        A 4x4 matrix representing an affine transform. 

    coords: ndarray 
        A 3xN array of 3d coordinates stored row-wise.  
    """
    XYZ = np.dot(affine[0:3,0:3], coords)
    XYZ[0,:] += affine[0,3]
    XYZ[1,:] += affine[1,3]
    XYZ[2,:] += affine[2,3]
    return XYZ 
