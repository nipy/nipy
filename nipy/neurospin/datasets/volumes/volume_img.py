"""
An image that stores the data as an (x, y, z, ...) array, with an
affine mapping to the world space
"""
import copy

import numpy as np
from scipy import ndimage

# Local imports
from ..transforms.affine_utils import to_matrix_vector, \
                from_matrix_vector
from ..transforms.affine_transform import AffineTransform
from ..transforms.transform import CompositionError

from .volume_grid import VolumeGrid

################################################################################
# class `VolumeImg`
################################################################################

class VolumeImg(VolumeGrid):
    """ A regularly-spaced image for embedding data in an x, y, z 3D
        world, for neuroimaging.

        This object is an ndarray representing a volume, with the first 3
        dimensions being spatial, and mapped to a named world space using
        an affine (4x4 matrix).

        Attributes
        ----------

        affine : 4x4 ndarray
            Affine mapping from indices to world coordinates.
        world_space : string
            Name of the world space the data is embedded in. For
            instance `mni152`.
        metadata : dictionnary
            Optional, user-defined, dictionnary used to carry around
            extra information about the data as it goes through
            transformations. The consistency of this information may not
            be maintained as the data is modified.
        interpolation : 'continuous' or 'nearest'
            String giving the interpolation logic used when calculating
            values in different world spaces
        _data : 
            Private pointer to the data.

        Notes
        ------

        The data is stored in an undefined way: prescalings might need to
        be applied to it before using it, or the data might be loaded on
        demand. The best practice to access the data is not to access the
        _data attribute, but to use the `get_data` method.
    """

    # most attributes are given by the VolumeField interface 

    #---------------------------------------------------------------------------
    # Attributes, VolumeImg interface
    #---------------------------------------------------------------------------

    # The affine (4x4 ndarray)
    affine = np.eye(4)

    #---------------------------------------------------------------------------
    # VolumeField interface
    #---------------------------------------------------------------------------

    def __init__(self, data, affine, world_space, metadata=None, 
                 interpolation='continuous'):
        """ Creates a new neuroimaging image with an affine mapping.

            Parameters
            ----------

            data : ndarray
                ndarray representing the data.
            affine : 4x4 ndarray
                affine transformation to the reference world space
            world_space : string
                name of the reference world space.
            metadata : dictionnary
                dictionnary of user-specified information to store with
                the image.
        """
        if not interpolation in ('continuous', 'nearest'):
            raise ValueError('interpolation must be either continuous '
                             'or nearest')
        self._data = data
        if not affine.shape == (4, 4):
            raise ValueError('The affine should be a 4x4 array')
        self.affine = affine
        self.world_space = world_space
        if metadata is None:
            metadata = dict()
        self.metadata = metadata
        self.interpolation = interpolation

    
    def like_from_data(self, data):
        # Use self.__class__ for subclassing.
        return self.__class__(data=data, 
                              affine=copy.copy(self.affine),
                              world_space=self.world_space,
                              metadata=copy.copy(self.metadata),
                              interpolation=self.interpolation,
                              )


    # Inherit docstring
    like_from_data.__doc__ = VolumeGrid.like_from_data.__doc__


    def get_transform(self):
        return AffineTransform('voxel_space', self.world_space, self.affine)


    # Inherit docstring
    get_transform.__doc__ = VolumeGrid.get_transform.__doc__


    def resampled_to_img(self, target_image, interpolation=None):
        if not target_image.world_space == self.world_space:
            raise CompositionError(
                'The two images are not embedded in the same world space')
        if isinstance(target_image, VolumeImg):
            return self.as_volume_img(affine=target_image.affine,
                                    shape=self.get_data().shape[:3],
                                    interpolation=interpolation)
        else:
            # IMPORTANT: Polymorphism can be implemented by walking the 
            # MRO and finding a method that does not raise
            # NotImplementedError. 
            return super(VolumeImg, self).resampled_to_img(target_image,
                                    interpolation=interpolation)


    # Inherit docstring
    resampled_to_img.__doc__ = VolumeGrid.resampled_to_img.__doc__


    def as_volume_img(self, affine=None, shape=None, 
                                        interpolation=None):
        if affine is None and shape is None:
            return copy.copy(self)
        if affine is None:
            affine = self.affine
        data = self.get_data()
        if shape is None:
            shape = data.shape[:3]
        shape = list(shape)
        if not len(shape) == 3:
            raise ValueError('The shape specified should be the shape '
                'the 3D grid, and thus of length 3. %s was specified'
                % shape )
        interpolation_order = self._get_interpolation_order(interpolation)
        if np.all(affine == self.affine):
            # Small trick to be more numericaly stable
            transform_affine = np.eye(4)
        else:
            transform_affine = np.dot(np.linalg.inv(self.affine), affine)
        A, b = to_matrix_vector(transform_affine)
        # For images with dimensions larger than 3D, pad A and b:
        n_dims = len(data.shape)
        if n_dims > 3:
            b = np.r_[b, np.zeros((n_dims - 3,))]
            A_ = np.eye(n_dims)
            A_[:3, :3] = A
            A = A_
            shape = shape + list(data.shape[3:])
        # If A is diagonal, ndimage.affine_transform is clever-enough 
        # to use a better algorithm
        if np.all(np.diag(np.diag(A)) == A):
           A = np.diag(A)
        resampled_data = ndimage.affine_transform(data, A,
                                            offset=b, 
                                            output_shape=shape,
                                            order=interpolation_order)
        return self.__class__(resampled_data, affine, 
                           self.world_space, metadata=self.metadata,
                           interpolation=self.interpolation)


    # Inherit docstring
    as_volume_img.__doc__ = VolumeGrid.as_volume_img.__doc__


    #---------------------------------------------------------------------------
    # VolumeImg interface
    #---------------------------------------------------------------------------

    def xyz_ordered(self):
        """ Returns an image with the affine diagonal and positive
            in the world space it is embedded in. 
        """
        A, b = to_matrix_vector(self.affine.copy())
        if not np.all((np.abs(A) > 0.001).sum(axis=0) == 1):
            raise CompositionError(
                'Cannot reorder the axis: the image affine contains rotations'
                )
        # Copy the image, we don't want to modify in place.
        img = self.__copy__()
        axis_numbers = np.argmax(np.abs(A), axis=0)
        while not np.all(np.sort(axis_numbers) == axis_numbers):
            first_inversion = np.argmax(np.diff(axis_numbers)<0)
            img = img._swapaxes(first_inversion+1, first_inversion)
            A, b = to_matrix_vector(img.affine)
            axis_numbers = np.argmax(np.abs(A), axis=0)

        # Now make sure the affine is positive
        pixdim = np.diag(A)
        data = img.get_data()
        if pixdim[0] < 0:
            b[0] = b[0] + pixdim[0]*(data.shape[0] - 1)
            pixdim[0] = -pixdim[0]
            data = data[::-1, ...]
        if pixdim[1] < 0:
            b[1] = b[1] + 1 + pixdim[1]*(data.shape[1] - 1)
            pixdim[1] = -pixdim[1]
            data = data[:, ::-1, ...]
        if pixdim[2] < 0:
            b[2] = b[2] + 1 + pixdim[2]*(data.shape[2] - 1)
            pixdim[2] = -pixdim[2]
            data = data[:, :, ::-1, ...]

        img._data = data
        img.affine = from_matrix_vector(np.diag(pixdim), b)
        return img
    

    def _swapaxes(self, axis1, axis2):
        """ Swap the axis axis1 and axis2 of the data array and reorder the 
            affine matrix to stay consistent with the data

            See also
            --------
            self.xyz_ordered
        """
        if (axis1 > 2) or (axis2 > 2):
            raise ValueError('Can swap axis only on spatial axis. '
                             'Use np.swapaxes of the data array.')
        reordered_data = np.swapaxes(self.get_data(), axis1, axis2)
        new_affine = self.affine
        order = np.array((0, 1, 2, 3))
        order[axis1] = axis2
        order[axis2] = axis1
        new_affine = new_affine.T[order].T
        return VolumeImg(reordered_data, new_affine, self.world_space, 
                                           metadata=self.metadata)

    #---------------------------------------------------------------------------
    # Private methods
    #---------------------------------------------------------------------------

    def _apply_transform(self, w2w_transform):
        """ Used for subclassing only. Do not call
        """
        new_v2w_transform = \
                        self.get_transform().composed_with(w2w_transform)
        if hasattr(new_v2w_transform, 'affine'):
            new_img = self.__class__(self.get_data(),
                                     new_v2w_transform.affine,
                                     new_v2w_transform.output_space,
                                     metadata=self.metadata,
                                     interpolation=self.interpolation)
        else:
            new_img = VolumeGrid(self.get_data(),
                                transform=new_v2w_transform,
                                metadata=self.metadata,
                                interpolation=self.interpolation)
        return new_img 


    def __repr__(self):
        options = np.get_printoptions()
        np.set_printoptions(precision=6, threshold=64, edgeitems=2)
        representation = \
                '%s(\n  data=%s,\n  affine=%s,\n  world_space=%s,\n  interpolation=%s)' % (
                self.__class__.__name__,
                '\n       '.join(repr(self._data).split('\n')),
                '\n         '.join(repr(self.affine).split('\n')),
                self.world_space,
                self.interpolation)
        np.set_printoptions(**options)
        return representation


    def __eq__(self, other):
        return (    isinstance(other, self.__class__)
                and np.all(self.get_data() == other.get_data())
                and np.all(self.affine == other.affine)
                and (self.world_space == other.world_space)
                and (self.interpolation == other.interpolation)
               )


