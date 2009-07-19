"""
The volume data class

This class represents indexable data embedded in a 3D space 
"""

import copy as copy

import numpy as np

# Local imports
from .volume_field import VolumeField
from ..transforms.transform import CompositionError


################################################################################
# class `VolumeData`
################################################################################

class VolumeData(VolumeField):
    """ A class representing data embedded in a 3D space 

        This object has data stored in an array like, that knows how it is 
        mapped to a 3D "real-world space", and how it can change real-world 
        coordinate system.

        **Attributes**

        :world_space: string 
            World space the data is embedded in. For instance `mni152`.

        :metadata: dictionnary
            Optional, user-defined, dictionnary used to carry around
            extra information about the data as it goes through
            transformations. The class does not garanty consistency
            of this information as the data is modified.

        :_data: 
            Private pointer to the data.

        **Notes**

        The data is stored in an undefined way: prescalings might need to
        be applied to it before using it, or the data might be loaded on
        demand. The best practice to access the data is not to access the
        _data attribute, but to use the `get_data` method.

        If the transform associated with the volume has no inverse
        mapping, data correspoding to a given world space position cannot
        be calulated. If it has no forward mapping, it is impossible to
        resample another dataset on the same support.
    """
    #---------------------------------------------------------------------------
    # Public attributes -- VolumeData interface
    #---------------------------------------------------------------------------

    # The interpolation logic used
    interpolation = 'continuous'

    #---------------------------------------------------------------------------
    # Private attributes -- VolumeData interface
    #---------------------------------------------------------------------------

    # The data (ndarray-like)
    _data = None

    #---------------------------------------------------------------------------
    # Public methods -- VolumeData interface
    #---------------------------------------------------------------------------


    def get_data(self):
        """ Return data as a numpy array.
        """
        return np.asanyarray(self._data)


    def like_from_data(self, data):
        """ Returns an volumetric data structure with the same
            relationship between data and world space, and same metadata, 
            but different data.

            Parameters
            -----------
            data: ndarray
        """
        raise NotImplementedError  


    def resampled_to_img(self, target_image, interpolation=None):
        """ Resample the data to be on the same voxel grid than the target 
            volume structure.

            Parameters
            ----------
            target_image : nipy image
                Nipy image onto the voxel grid of which the data will be
                resampled.
            interpolation : None, 'continuous' or 'nearest', optional
                Interpolation type used when calculating values in
                different word spaces. If None, the image's interpolation
                logic is used.

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
                "The two images are not embedded in the same world space")
        my_v2w_transform = self.get_transform()
        x, y, z = target_image.get_world_coords()
        new_data = self.values_in_world(x, y, z, 
                                        interpolation=interpolation)
        new_img = target_image.like_from_data(new_data) 
        new_img.metadata = copy.copy(self.metadata)
        return new_img


    #---------------------------------------------------------------------------
    # Private methods
    #---------------------------------------------------------------------------

    def _apply_transform(self, w2w_transform):
        """ Method applying the transform: inner part of
            transformed_with, used in subclassing.
        """
        new_v2w_transform = \
                        self.get_transform().composed_with(w2w_transform)
        new_img = copy.copy(self)
        new_img._transform = new_v2w_transform
        return new_img


    def _get_interpolation_order(self, interpolation):
        """ Inner method used to get the interpolation type for the
            image.
        """
        if interpolation is None:
            interpolation = self.interpolation
        if interpolation == 'continuous':
            interpolation_order = 3
        elif interpolation == 'nearest':
            interpolation_order = 0
        else:
            raise ValueError("interpolation must be either 'continuous' "
                             "or 'nearest'")
        return interpolation_order

    # TODO: We need to implement (or check if implemented) hashing,
    # weakref, pickling? 
        

    def __repr__(self):
        options = np.get_printoptions()
        np.set_printoptions(precision=6, threshold=64, edgeitems=2)
        representation = \
                '%s(\n  data=%s,\n  world_space=%s,\n  interpolation=%s)' % (
                self.__class__.__name__,
                '\n       '.join(repr(self._data).split('\n')),
                self.world_space,
                self.interpolation,
                )
        np.set_printoptions(**options)
        return representation


    def __copy__(self):
        return self.like_from_data(self.get_data().copy())


    def __deepcopy__(self, option):
        """ Copy the Volume and the arrays and metadata it contains.
        """
        out = self.__copy__()
        out.metadata = copy.deepcopy(self.metadata)
        return out


    def __eq__(self, other):
        return (    self.world_space       == other.world_space 
                and self.get_transform()   == other.get_transform()
                and np.all(self.get_data() == other.get_data())
                and self.interpolation     == other.interpolation
               )

