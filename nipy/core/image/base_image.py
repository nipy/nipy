"""
The base image interface.

This defines the nipy image interface.
"""

################################################################################
# class `BaseImage`
################################################################################

class BaseImage(object):
    """ The base image for neuroimaging.

        This object represents numerical values embedded in a
        3-dimensional world space (called a field in physics and
        engineering)
        
        This is an abstract base class: it defines the interface, but not
        the logics.

        **Attributes**

        :world_space: string 
            World space the data is embedded in. For instance `mni152`.

        :metadata: dictionnary
            Optional, user-defined, dictionnary used to carry around
            extra information about the data as it goes through
            transformations. The Image class does not garanty consistency
            of this information as the data is modified.

    """
    #---------------------------------------------------------------------------
    # Public attributes -- BaseImage interface
    #---------------------------------------------------------------------------

    # The name of the reference coordinate system
    world_space = ''

    # User defined meta data
    metadata = dict()

    #---------------------------------------------------------------------------
    # Public methods -- BaseImage interface
    #---------------------------------------------------------------------------

    def get_transform(self):
        """ Returns the transform object associated with the image which is a 
            general description of the mapping from the voxel space to the 
            world space.
            
            Returns
            -------
            transform : nipy.core.Transform object
        """
        raise NotImplementedError 


    def resampled_to_img(self, target_image, interpolation=None):
        """ Resample the image to be on the same voxel grid than the target 
            image.

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
        raise NotImplementedError


    # XXX: rename to as_volume_image
    def resampled_to_grid(self, affine=None, shape=None, interpolation=None):
        """ Resample the image to be an image with the data points lying
            on a regular grid with an affine mapping to the word space.

            Parameters
            ----------
            affine : 4x4 ndarray
                Affine of the new voxel grid or transform object pointing
                to the new voxel coordinate grid. If a 3x3 ndarray is given, 
                it is considered to be the rotation part of the affine, 
                and the best possible bounding box is calculated.
            interpolation : None, 'continuous' or 'nearest', optional
                Interpolation type used when calculating values in
                different word spaces. If None, the image's interpolation
                logic is used.

            Returns
            -------
            resampled_image : nipy XYZImage
                New nipy image with the data resampled in the given
                affine.

            Notes
            -----
            The coordinate system of the image is not changed: the
            returned image points to the same world space.
        """
        raise NotImplementedError


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
                different word spaces. If None, the image's interpolation
                logic is used.

            Returns
            -------
            values : number or ndarray
                Data values interpolated at the given world position.
                This is a number or an ndarray, depending on the shape of
                the input coordinate.
        """
        raise NotImplementedError


    # XXX: rename to composed_with_transform
    def transformed_with(self, w2w_transform):
        """ Return a new image embedding the same data in a different 
            word space using the given world to world transform.

            Parameters
            ----------
            w2w_transform : transform object
                The transform object giving the mapping between
                the current world space of the image, and the new
                word space.

            Returns
            --------
            remapped_image : nipy image
                An image containing the same data, expressed
                in the new world space.

        """
        raise NotImplementedError

    #---------------------------------------------------------------------------
    # Private methods
    #---------------------------------------------------------------------------

    # The subclasses should implement __repr__, __copy__, __deepcopy__,
    # __eq__ 
    # TODO: We need to implement (or check if implemented) hashing,
    # weakref, pickling? 
        

