# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The base volumetric field interface 

This defines the nipy volumetric structure interface.
"""
from ..transforms.transform import CompositionError

################################################################################
# class `VolumeField`
################################################################################

class VolumeField(object):
    """ The base volumetric structure.

        This object represents numerical values embedded in a
        3-dimensional world space (called a field in physics and
        engineering)
        
        This is an abstract base class: it defines the interface, but not
        the logics.

        Attributes
        ----------

        world_space: string 
            World space the data is embedded in. For instance `mni152`.
        metadata: dictionnary
            Optional, user-defined, dictionnary used to carry around
            extra information about the data as it goes through
            transformations. The consistency of this information is not
            maintained as the data is modified.

    """
    #---------------------------------------------------------------------------
    # Public attributes -- VolumeField interface
    #---------------------------------------------------------------------------

    # The name of the reference coordinate system
    world_space = ''

    # User defined meta data
    metadata = dict()

    #---------------------------------------------------------------------------
    # Public methods -- VolumeField interface
    #---------------------------------------------------------------------------

    def get_transform(self):
        """ Returns the transform object associated with the volumetric
            structure which is a general description of the mapping from 
            the values to the world space.
            
            Returns
            -------
            transform : nipy.datasets.Transform object
        """
        raise NotImplementedError 


    def resampled_to_img(self, target_image, interpolation=None):
        """ Resample the volume to be sampled similarly than the target 
            volumetric structure.

            Parameters
            ----------
            target_image : nipy volume 
                Nipy volume structure onto the grid of which the data will be
                resampled.
            interpolation : None, 'continuous' or 'nearest', optional
                Interpolation type used when calculating values in
                different word spaces. If None, the volume's interpolation
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
        # IMPORTANT: Polymorphism can be implemented by walking the 
        # MRO and finding a method that does not raise
        # NotImplementedError. 
        raise NotImplementedError


    def as_volume_img(self, affine=None, shape=None,
                        interpolation=None, copy=True):
        """ Resample the image to be an image with the data points lying
            on a regular grid with an affine mapping to the word space (a
            nipy VolumeImg).

            Parameters
            ----------
            affine: 4x4 or 3x3 ndarray, optional
                Affine of the new voxel grid or transform object pointing
                to the new voxel coordinate grid. If a 3x3 ndarray is given, 
                it is considered to be the rotation part of the affine, 
                and the best possible bounding box is calculated,
                in this case, the shape argument is not used. If None
                is given, a default affine is provided by the image.
            shape: (n_x, n_y, n_z), tuple of integers, optional
                The shape of the grid used for sampling, if None
                is given, a default affine is provided by the image.
            interpolation : None, 'continuous' or 'nearest', optional
                Interpolation type used when calculating values in
                different word spaces. If None, the image's interpolation
                logic is used.

            Returns
            -------
            resampled_image : nipy VolumeImg
                New nipy VolumeImg with the data sampled on the grid
                defined by the affine and shape.

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


    def composed_with_transform(self, w2w_transform):
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
        if not w2w_transform.input_space == self.world_space:
            raise CompositionError(
                "The transform given does not apply to "
                "the image's world space:\n%s\n\n%s" % 
                (w2w_transform, self)
                )
        new_img = self._apply_transform(w2w_transform)
        new_img.world_space = w2w_transform.output_space
        return new_img 


    #---------------------------------------------------------------------------
    # Private methods
    #---------------------------------------------------------------------------

    # The subclasses should implement __repr__, __copy__, __deepcopy__,
    # __eq__ 
    # TODO: We need to implement (or check if implemented) hashing,
    # weakref, pickling? 
        
    def _apply_transform(self, w2w_transform):
        """ Implement this method to put in the logic of applying a
            transformation on the image class.
        """
        raise NotImplementedError
