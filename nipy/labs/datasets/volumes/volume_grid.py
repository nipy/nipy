# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The volume grid class. 

This class represents data lying on a (non rigid, non regular) grid embedded 
in a 3D world represented as a 3+D array.
"""

import copy as copy

import numpy as np
from scipy import ndimage

# Local imports
from .volume_data import VolumeData
from ..transforms.affine_utils import apply_affine, from_matrix_vector


################################################################################
# class `VolumeGrid`
################################################################################

class VolumeGrid(VolumeData):
    """ A class representing data stored in a 3+D array embedded in a 3D
        world.

        This object has data stored in an array-like multidimensional 
        indexable objects, with the 3 first dimensions corresponding to
        spatial axis and defining a 3D grid that may be non-regular or
        non-rigid.

        The object knows how the data is mapped to a 3D "real-world
        space", and how it can change real-world coordinate system. The
        transform mapping it to world is arbitrary, and thus the grid
        can be warped: in the world space, the grid may not be regular or
        orthogonal.

        Attributes
        -----------

        world_space: string 
            World space the data is embedded in. For instance `mni152`.

        metadata: dictionnary
            Optional, user-defined, dictionnary used to carry around
            extra information about the data as it goes through
            transformations. The consistency of this information is not
            maintained as the data is modified.

        _data: 
            Private pointer to the data.

        Notes
        ------

        The data is stored in an undefined way: prescalings might need to
        be applied to it before using it, or the data might be loaded on
        demand. The best practice to access the data is not to access the
        _data attribute, but to use the `get_data` method.

        If the transform associated with the image has no inverse
        mapping, data corresponding to a given world space position cannot
        be calulated. If it has no forward mapping, it is impossible to
        resample another dataset on the same support.
    """
    #---------------------------------------------------------------------------
    # Public methods -- VolumeGrid interface
    #---------------------------------------------------------------------------

    def __init__(self, data, transform, metadata=None,
                 interpolation='continuous'):
        """ The base image containing data.

            Parameters
            ----------

            data: ndarray
                n dimensional array giving the embedded data, with the 3
                first dimensions being spatial.
            transform: nipy transform object
                The transformation from voxel to world.
            metadata : dictionnary, optional
                Dictionnary of user-specified information to store with
                the image.
            interpolation : 'continuous' or 'nearest', optional
                Interpolation type used when calculating values in
                different word spaces.
        """
        if not interpolation in ('continuous', 'nearest'):
            raise ValueError('interpolation must be either continuous '
                             'or nearest')
        self._data       = data
        self._transform  = transform
        self.world_space = transform.output_space
        if metadata is None:
            metadata = dict()
        self.metadata = metadata
        self.interpolation = interpolation


    def as_volume_img(self, affine=None, shape=None, 
                                        interpolation=None, copy=True):
        if affine is None:
            affine = np.eye(3)
        if affine.shape[0] == 3 or shape is None:
            affine3d = affine[:3, :3]
            affine4d = np.eye(4)
            affine4d[:3, :3] = affine3d
            x, y, z = self.get_world_coords()
            x, y, z = apply_affine(x, y, z, np.linalg.inv(affine4d))
            xmin = x.min()
            ymin = y.min()
            zmin = z.min()
            if affine.shape[0] == 3:
                offset = np.array((xmin, ymin, zmin))
                offset = np.dot(affine3d, offset)
                affine = from_matrix_vector(affine3d, offset[:3])
            if shape is None:
                xmax = x.max()
                ymax = y.max()
                zmax = z.max()
                shape = (np.ceil(xmax - xmin)+1,
                         np.ceil(ymax - ymin)+1,
                         np.ceil(zmax - zmin)+1, )
        shape = list(shape)
        if not len(shape) == 3:
            raise ValueError('The shape specified should be the shape '
                'the 3D grid, and thus of length 3. %s was specified'
                % shape )
        x, y, z = np.indices(shape)
        x, y, z = apply_affine(x, y, z, affine)
        values = self.values_in_world(x, y, z)
        # We import late to avoid circular import
        from .volume_img import VolumeImg
        return VolumeImg(values, affine, 
                           self.world_space, metadata=self.metadata,
                           interpolation=self.interpolation)


    # Inherit docstring
    as_volume_img.__doc__ = VolumeData.as_volume_img.__doc__


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

    # Inherit docstring
    like_from_data.__doc__ = VolumeData.like_from_data.__doc__


    def get_transform(self):
        """ Returns the transform object associated with the image which is a 
            general description of the mapping from the voxel space to the 
            world space.
            
            Returns
            -------
            transform : nipy.core.Transform object
        """
        return self._transform


    # Inherit docstring
    get_transform.__doc__ = VolumeData.get_transform.__doc__


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
        coords = np.c_[i, j, k].T
        # work round an ndimage deficiency in scipy <= 0.9.0.
        # See: https://github.com/scipy/scipy/pull/64
        if coords.dtype == np.dtype(np.intp):
            coords = coords.astype(np.dtype(coords.dtype.str))
        data = self.get_data()
        data_shape = list(data.shape)
        n_dims = len(data_shape)
        if n_dims > 3:
            # Iter in a set of 3D volumes, as the interpolation problem is 
            # separable in the extra dimensions. This reduces the
            # computational cost
            data = np.reshape(data, data_shape[:3] + [-1])
            data = np.rollaxis(data, 3)
            values = [ ndimage.map_coordinates(slice, coords,
                                               order=interpolation_order)
                       for slice in data]
            values = np.array(values)
            values = np.swapaxes(values, 0, -1)
            values = np.reshape(values, shape + data_shape[3:])
        else:
            values = ndimage.map_coordinates(data, coords,
                                        order=interpolation_order)
            values = np.reshape(values, shape)
        return values

    # Inherit docstring
    values_in_world.__doc__ = VolumeData.values_in_world.__doc__

