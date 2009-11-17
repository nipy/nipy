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

    spacing: tuple of ints, optional
        Subsampling factors 

    affine: ndarray 
        4x4 affine matrix that maps grid coordinates to desired coordinates. 
    
    """
    def __init__(self, shape, corner=(0,0,0), spacing=(1,1,1), affine=None):

        if not len(shape) == 3:
            raise ValueError('The specified shape should be of length 3.')
        if not len(corner) == 3:
            raise ValueError('The specified corner should be of length 3.')
        if not len(spacing) == 3:
            raise ValueError('The specified subsampling factors should be of length 3.')
        self._shape = np.array(shape[0:3], dtype='uint')
        self._corner = np.array(corner[0:3], dtype='uint')
        self._spacing = np.array(spacing[0:3], dtype='uint')
        self._affine = affine

    def _get_shape(self):
        return self._shape

    def _get_corner(self):
        return self._corner

    def _get_spacing(self):
        return self._spacing
    
    def coords(self):
        c = self._corner
        s = self._shape
        sp = self._spacing
        x,y,z = np.mgrid[c[0]:c[0]+s[0]:sp[0],
                         c[1]:c[1]+s[1]:sp[1],
                         c[2]:c[2]+s[2]:sp[2]]
        x,y,z = x.ravel(),y.ravel(),z.ravel()
        if self._affine == None: 
            return xyz
        return apply_affine(self._affine, xyz)
                   
    def slice(self, array): 
        c = self._corner
        s = self._shape
        sp = self._spacing
        return array[c[0]:c[0]+s[0]:sp[0],
                     c[1]:c[1]+s[1]:sp[1],
                     c[2]:c[2]+s[2]:sp[2]]

    def __str__(self):
        return 'Grid: shape %s, corner %s, spacing %s' % (self._shape, self._corner, self._spacing) 

    shape = property(_get_shape)
    corner = property(_get_corner)
    spacing = property(_get_spacing)
    

# class `Block`

class Block(object): 
    """ A basic class to represent an image defined on a regular
    lattice.
    """
    def __init__(self, data, affine, world=None, cval=0): 
        self._data = data
        self._affine = affine
        self._world = world 
        self._cval = cval 

    def _get_shape(self):
        return self._data._shape

    def _get_dtype(self): 
        return self._data.dtype

    def _get_affine(self):
        return self._affine

    def _get_data(self):
        """
        Get a 3d array.
        """
        return self._data

    def values(self, coords, world_coords=False, order=1, optimize=True):

        # If coords are intended to describe world coordinates,
        # immediately convert them into grid coordinates
        if world_coords:
            to_grid = np.linalg.inv(self._affine)
            if isinstance(coords, Grid): 
                # compose transforms
                if coords._affine: 
                    coords._affine = np.dot(to_grid, self._affine)
                else:
                    coords._affine = to_grid
            else: 
                coords = apply_affine(to_grid, coords)

              
        # If coords is a Grid instance, we do not need to explicitely
        # compute the coordinates
        if isinstance(coords, Grid): 
            if coords._affine:
                return ndimage.affine_transform(self._data, 
                                                coords._affine[0:3,0:3],
                                                offset=coords._affine[0:3,3],
                                                order=order, 
                                                cval=self._cval)
            else:
                return coords.slice(self._data)
        
        # At this point, we know that coords is an explicit tuple of
        # coordinates 

        # Avoid interpolation if coordinates are integers
        if [c.dtype==int for c in coords].count(True):
            return self._data[coords]
        else: # interpolation needed
            return ndimage.map_coordinates(self._data, 
                                           np.c_[coords].T, 
                                           order=order, 
                                           cval=self._cval)

    shape = property(_get_shape)
    dtype = property(_get_dtype)
    affine = property(_get_affine)
    data = property(_get_data)

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
    def __init__(self, data, affine, mask=None, world=None, cval=0):
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
            
            cval: number, optional 
                Background value (outside image boundaries).  

        """

        # TODO: Check that the mask array, if provided, is consistent
        # with the data array
        # Check array datatype and cval 
        if mask == None: 
            self._masked = False
            mask = Grid(data.shape)
            self._data = None
            self._block = Block(data, affine, world=world, cval=cval)
        elif isinstance(mask, Grid):
            self._masked = True
            self._data = None
            # block to world affine transformation
            block2im_affine = np.diag(np.concatenate((mask._spacing,[1]),1))
            block2im_affine[0:3,3] = mask._corner
            block_affine = np.dot(affine, block2im_affine)
            self._block = Block(mask.slice(data), block_affine, world=world, cval=cval)
        else:
            self._masked = True
            self._data = data[mask]
            self._block = None 
        self._shape = data.shape
        self._dtype = data.dtype
        self._mask = mask 
        self._affine = affine
        self._world = world
        self._cval = cval

    def _get_masked(self):
        return self._masked

    def _get_shape(self):
        return self._shape

    def _get_dtype(self):
        return self._dtype

    def _get_affine(self):
        return self._affine

    def _get_data(self):
        """
        Get a 3d array.
        """
        if self._masked:
            data = np.zeros(self._shape, dtype=self._dtype)
            if not self._cval == 0:  
                data += self._cval
            if self._block: # mask is a Grid instance
                c = self._mask._corner 
                s = self._mask._shape
                sp = self._mask._spacing
                data[c[0]:c[0]+s[0]:sp[0],
                     c[1]:c[1]+s[1]:sp[1],
                     c[2]:c[2]+s[2]:sp[2]] = self._block._data
            else:
                data[self._mask] = self._data
            return data 
        else:
            return self._block._data

    def crop(self): 
        """
        Define a bounding box. 
        """
        if self._block:
            data = self._block._data
            affine = self._block._affine
            return Image(data, 
                         affine, 
                         world=self._world, 
                         cval=self._cval)
        

        # define a bounding box
        corner = np.asarray([self._mask[i].min() for i in range(3)])
        shape = [1+self._mask[i].max()-corner[i] for i in range(3)]
        data = np.zeros(shape, dtype=self._dtype)
        if not self._cval == 0:  
            data += self._cval
        data[[self._mask[i]-corner[i] for i in range(3)]] = self._data
        box2im_affine = np.eye(4)
        box2im_affine[0:3,3] = corner
        affine = np.dot(self._affine, box2im_affine)
        return Image(data, 
                     affine, 
                     world=self._world, 
                     cval=self._cval)

        """
        we need to have: 
          corner <= mask < corner + shape 
        """
        
        """
        if self._masked:
            array = np.zeros(shape, dtype=self._data.dtype)
            if not self._cval == 0:  
                array += self._cval

            tmp = (np.c_[self._mask]-corner).T
            submask = np.where(np.all(tmp>0, axis=0))

            if isinstance(self._mask, Grid):
                c = mask._corner - corner
                s = mask._shape
                sp = mask._spacing
                array[c[0]:c[0]+s[0]:sp[0],
                      c[1]:c[1]+s[1]:sp[1],
                      c[2]:c[2]+s[2]:sp[2]] = self._data
            else:
                array[mask - corner] = self._data[submask]
            return array
        else:
            return self._data[corner[0]:corner[0]+shape[0]:spacing[0], 
                              corner[1]:corner[1]+shape[1]:spacing[1], 
                              corner[2]:corner[2]+shape[2]:spacing[2]]

        """
    
    def values(self, coords, world_coords=False, order=1, optimize=True):
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

        """
        Try to avoid interpolation whenever possible. 
        If coords is included in the image mask, no interpolation needed. 
        Otherwise, interpolation. Then, we have two cases: 
          - coords is a grid: we can call an interpolation routine that does 
            not require an explicit list of points
          - coords is not a grid: we have to call another function. 

        Testing mask inclusion:
          If mask is a grid: 
            - If coords is a grid, then easy
            - If coords is not, then it's not supposed to fit in a
              grid (otherwise, the user should have defined it as a grid)
          
          If mask is not a Grid:
            then we have to go for a element-wise comparison whether 
            or not coords is a grid
        """

        if self._block: 
            block = self._block
        else: 
            block = Block(self.get_data(), self._affine)
        return block.values(coords, world_coords, order, optimize)
    
    
    
        # If coords are intended to describe world coordinates,
        # immediately convert them into grid coordinates
        """if world_coords:
            to_grid = np.linalg.inv(self._affine)
            if isinstance(coords, Grid): 
                # compose transforms
                if coords._affine: 
                    coords._affine = np.dot(to_grid, self._affine)
                else:
                    coords._affine = to_grid
            else: 
                coords = apply_affine(to_grid, coords)

        # Get the relevant 3d array data
        if self._masked: 
            array, corner = bounding_box(self._data, self._mask, 
                                         background=self._cval)
        else: 
            array = self._data
            corner = np.zeros(3, dtype='uint')
            
        # If coords is a Grid instance, we do not need to explicitely
        # compute the coordinates
        if isinstance(coords, Grid): 
            if coords._affine:
                return ndimage.affine_transform(array, 
                                                coords._affine[0:3,0:3],
                                                offset=coords._affine[0:3,3]-corner,
                                                order=order)
            else:
                return array[coords.coords()-corner]
        
        # At this point, we know that coords is an explicit tuple of
        # coordinates 

        # Avoid interpolation if coordinates are integers
        if [c.dtype==int for c in coords].count(True):
            array, corner = bounding_box(self._data, self._mask, 
                                         background=self._cval)

            return None
        else: # interpolation needed
            return ndimage.map_coordinates(self.get_data(), 
                                           np.c_[coords].T, 
                                           order=order)
        """

    masked = property(_get_masked)
    shape = property(_get_shape)
    dtype = property(_get_dtype)
    affine = property(_get_affine)
    data = property(_get_data)


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


def apply_affine(affine, XYZ):
    """
    Parameters
    ----------
    affine: ndarray 
        A 4x4 matrix representing an affine transform. 

    coords: tuple of ndarrays 
        tuple of 3d coordinates stored row-wise: (X,Y,Z)  
    """
    trans_XYZ = np.array(XYZ)
    trans_XYZ[:] = np.dot(affine[0:3,0:3], trans_XYZ[:])
    trans_XYZ[0,:] += affine[0,3]
    trans_XYZ[1,:] += affine[1,3]
    trans_XYZ[2,:] += affine[2,3]
    return tuple(trans_XYZ)


def bounding_box(data, mask, shape=None, background=0): 
    """ 
    Return a 3d array. The mask coordinates need to be in the range
    [corner, shape+corner[. 
    """
    if shape == None: 
        corner = np.asarray([self._mask[i].min() for i in (0,1,2)])
        shape = [1+self._mask[i].max()-corner[i] for i in (0,1,2)]
    else: 
        corner = np.zeros(3, dtype='uint')
    array = np.zeros(shape, dtype=data.dtype)
    if not background == 0:  
        array += background
    if isinstance(mask, Grid):
        c = mask._corner - corner
        s = mask._shape
        sp = mask._spacing
        array[c[0]:c[0]+s[0]:sp[0],
              c[1]:c[1]+s[1]:sp[1],
              c[2]:c[2]+s[2]:sp[2]] = data
    else:
        array[mask - corner] = data
    return array, corner
