"""
The base image interface.
"""
import copy

import numpy as np
from scipy import ndimage

# Local imports
from ..transforms.affine_utils import to_matrix_vector
from ..transforms.transform import CompositionError
from ..transforms.affine_transform import AffineTransform

from .base_image import BaseImage

################################################################################
# class `AffineImage`
################################################################################

class AffineImage(BaseImage):
    """ The affine image for neuroimaging.

        This object is nothing more than an ndarray representing a
        volume, with the first 3 dimensions being spatial, and mapped to 
        a named world space using an affine (4x4 matrix).

        **Attributes**

        :affine: 4x4 ndarray

            Affine mapping from indices to world coordinates.

        :world_space: string

            Name of the world space the data is embedded in. For
            instance `mni152`.

        :metadata: dictionnary

            Optional, user-defined, dictionnary used to carry around
            extra information about the data as it goes through
            transformations. The Image class does not garanty consistency
            of this information as the data is modified.

        :_data: 

            Private pointer to the data.

        **Notes**

        The data is stored in an undefined way: prescalings might need to
        be applied to it before using it, or the data might be loaded on
        demand. The best practice to access the data is not to access the
        _data attribute, but to use the `get_data` method.
    """

    #---------------------------------------------------------------------------
    # Attributes, BaseImage interface
    #---------------------------------------------------------------------------

    # The name of the world space the image is embedded in
    world_space = ''

    # User defined meta data
    metadata = dict()

    # The data (ndarray)
    _data = None

    # XXX: Need an attribute to determine in a clever way the
    # interplation order/method

    #---------------------------------------------------------------------------
    # Attributes, AffineImage interface
    #---------------------------------------------------------------------------

    # The affine (4x4 ndarray)
    affine = np.eye(4)

    #---------------------------------------------------------------------------
    # BaseImage interface
    #---------------------------------------------------------------------------

    def __init__(self, data, affine, world_space, metadata=None):
        """ Creates a new neuroimaging image with an affine mapping.

            Parameters
            ----------

            data : ndarray
                ndarray representing the data.
            affine : 4x4 ndarray
                affine transformation to the reference world space
            world_space : string
                name of the reference world space.
        """
        self._data = data
        if not affine.shape == (4, 4):
            raise ValueError('The affine should be a 4x4 array')
        self.affine = affine
        self.world_space = world_space
        if metadata is None:
            metadata = dict()
        self.metadata = metadata

    
    def get_lookalike(self, data):
        """ Returns an image with the same transformation and metadata,
            but different data.

            Parameters
            -----------
            data: ndarray
        """
        # Use self.__class__ for subclassing.
        return self.__class__(data=data, 
                              affine=copy.copy(self.affine),
                              world_space=self.world_space,
                              )


    def get_transform(self):
        """ Returns the transform object associated with the image which is a 
            general description of the mapping from the voxel grid to the 
            world space.

            Returns
            -------
            transform : nipy.core.Transform object
        """
        return AffineTransform(self.affine, 'voxel_space', self.world_space)


    def resampled_to_affine(self, new_affine, interpolation_order=3):
        """ Resample the image to be an affine image.

            Parameters
            ----------
            new_affine : 4x4 ndarray
                Affine of the new grid or transform object pointing
                to the new grid. 
            interpolation_order : int, optional
                Order of the spline interplation. If 0, nearest-neighboor 
                interpolation is performed.

            Returns
            -------
            resampled_image : nipy AffineImage
                New nipy image with the data resampled in the given
                affine.

            Notes
            -----
            The world space of the image is not changed: the
            returned image points to the same world space.
        """
        if np.all(new_affine == self.affine):
            # Small trick to be more numericaly stable
            # XXX: The trick should be implemented to work for
            # affine.shape = (3, 3)
            transform_affine = np.eye(4)
        else:
            transform_affine = np.dot(np.linalg.inv(self.affine), new_affine)
        A, b = to_matrix_vector(transform_affine)
        # If A is diagonal, ndimage.affine_transform is clever-enough 
        # to use a better algorithm
        if np.all(np.diag(np.diag(A)) == A):
           A = np.diag(A)
        data = self.get_data()
        resampled_data = ndimage.affine_transform(data, A,
                                        offset=b, 
                                        output_shape=data.shape,
                                        order=interpolation_order)
        return AffineImage(resampled_data, new_affine, 
                           self.world_space, metadata=self.metadata)


    def resampled_to_img(self, target_image, interpolation_order=3):
        """ Resample the image to be on the same grid than the target image.

            Parameters
            ----------
            target_image : nipy image
                Nipy image onto the grid of which the data will be
                resampled.
            interpolation_order : int, optional
                Order of the spline interplation. If 0, nearest neighboor 
                interpolation is performed.

            Returns
            -------
            resampled_image : nipy_image
                New nipy image with the data resampled.

            Notes
            -----
            Both the target image and the original image should be
            embedded in the same world space.
        """
        if not target_image.world_space == self.world_space:
            raise CompositionError(
                'The two images are not embedded in the same world space')
        target_shape = target_image.get_data().shape[:3]
        if hasattr(target_image, 'affine'):
            new_im = self.resampled_to_affine(target_image.affine,
                                    interpolation_order=interpolation_order)
            # Some massaging to get the shape right
            new_data = np.zeros(target_image.get_data().shape)
            x_new, y_new, z_new = new_data.shape
            old_data = new_im.get_data() 
            x_old, y_old, z_old = old_data.shape
            new_data[:min(x_old, x_new), :min(y_old, y_new), 
                     :min(z_old, z_new)] = \
                    old_data[:min(x_old, x_new), :min(y_old, y_new), 
                            :min(z_old, z_new)]
            new_im._data = new_data
            return new_im
        else:
            # XXX: we need a dispatcher pattern or to encode the
            # information in the transform
            return super(AffineImage, self).resampled_to_img(target_image,
                                    interpolation_order=interpolation_order)

    
    #---------------------------------------------------------------------------
    # AffineImage interface
    #---------------------------------------------------------------------------

    def xyz_ordered(self):
        """ Returns an image with the affine diagonal and positive
            in the world space it is embedded in. 
        """
        A, b = to_matrix_vector(self.affine)
        if not np.all((np.abs(A) > 0.001).sum(axis=0) == 1):
            raise CompositionError(
                'Cannot reorder the axis: the image affine contains rotations'
                )
        img = self
        axis_numbers = np.argmax(np.abs(A), axis=0)
        while not np.all(np.sort(axis_numbers) == axis_numbers):
            first_inversion = np.argmax(np.diff(axis_numbers)<0)
            img = img._swapaxes(first_inversion+1, first_inversion)
            A, b = to_matrix_vector(img.affine)
            axis_numbers = np.argmax(np.abs(A), axis=0)
            # XXX: the affine is not yet positive.
        return img
    
    
    def _swapaxes(self, axis1, axis2):
        """ Swap the axis axis1 and axis2 of the data array and reorder the 
            affine matrix to stay consistent with the data

            See also
            --------
            self.xyz_ordered
        """
        if (   (axis1 > 2) and (axis2 < 2)
                                or (axis2 > 2) and (axis1 < 2)):
            raise ValueError('Cannot swap a spatial axis with a non'
                             'spatial axis')
        reordered_data = np.swapaxes(self.get_data(), axis1, axis2)
        new_affine = self.affine
        if axis1 < 3:
            order = np.array((0, 1, 2, 3))
            order[axis1] = axis2
            order[axis2] = axis1
            new_affine = new_affine.T[order].T
        return AffineImage(reordered_data, new_affine, self.world_space, 
                                           metadata=self.metadata)

    #---------------------------------------------------------------------------
    # Private methods
    #---------------------------------------------------------------------------
    
    def __repr__(self):
        options = np.get_printoptions()
        np.set_printoptions(precision=6, threshold=64, edgeitems=2)
        representation = \
                '%s(\n  data=%s,\n  affine=%s,\n  world_space=%s)' % (
                self.__class__.__name__,
                '\n       '.join(repr(self._data).split('\n')),
                '\n         '.join(repr(self.affine).split('\n')),
                self.world_space)
        np.set_printoptions(**options)
        return representation


    def __copy__(self):
        """ Copy the Image and the arrays and metadata it contains.
        """
        return self.__class__(data=self.get_data().copy(), 
                              affine=self.affine.copy(),
                              world_space=self.world_space,
                              metadata=self.metadata.copy())


    def __deepcopy__(self, option):
        """ Copy the Image and the arrays and metadata it contains.
        """
        import copy
        return self.__class__(data=self.get_data().copy(), 
                              affine=self.affine.copy(),
                              world_space=self.world_space,
                              metadata=copy.deepcopy(self.metadata))


    def __eq__(self, other):
        return (    isinstance(other, self.__class__)
                and np.all(self.get_data() == other.get_data())
                and np.all(self.affine == other.affine)
                and (self.world_space == other.world_space))


