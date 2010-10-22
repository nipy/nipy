# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The base image interface.
"""

import numpy as np
from scipy import ndimage

# Local imports
from .image import Image
from ..transforms.affines import to_matrix_vector
from ..reference.coordinate_system import CoordinateSystem
from ..reference.coordinate_map import (AffineTransform,
                                        product as cmap_product)

################################################################################
# class `AffineImage`
################################################################################

class AffineImage(Image):

    """ The affine image for nipy.

        This object is a subclass of Image that
        assumes the first 3 coordinates
        are spatial. 

        **Attributes**

        :metadata: dictionnary

            Optional, user-defined, dictionnary used to carry around
            extra information about the data as it goes through
            transformations. The Image class does not garanty consistency
            of this information as the data is modified.

        :_data: 

            Private pointer to the data.

        **Properties**

        :affine: 4x4 ndarray

            Affine mapping from voxel axes to world coordinates
            (world coordinates are always forced to be 'x', 'y', 'z').

        :spatial_coordmap: AffineTransform

            Coordinate map describing the spatial coordinates 
            (always forced to be 'x', 'y', 'z') and the coordinate
            axes with names axis_names[:3].
           
        :coordmap: AffineTransform

            Coordinate map describing the relationship between
            all coordinates and axis_names.


        **Notes**

        The data is stored in an undefined way: prescalings might need to
        be applied to it before using it, or the data might be loaded on
        demand. The best practice to access the data is not to access the
        _data attribute, but to use the `get_data` method.
    """


    #---------------------------------------------------------------------------
    # Attributes, BaseImage interface
    #---------------------------------------------------------------------------

    # The name of the reference coordinate system
    coord_sys = ''

    # User defined meta data
    metadata = dict()

    # The data (ndarray)
    _data = None

    # XXX: Need an attribute to determine in a clever way the
    # interplation order/method

    def __init__(self, data, affine, coord_sys, metadata=None):
        """ Creates a new nipy image with an affine mapping.

        Parameters
        ----------
        data : ndarray
            ndarray representing the data.
        affine : 4x4 ndarray
            affine transformation to the reference coordinate system
        coord_system : string
            name of the reference coordinate system.
        """
        affine = np.asarray(affine)
        if affine.shape != (4,4):
            raise ValueError('Affine image takes 4x4 affine as input')
        function_domain = CoordinateSystem(['axis%d' % i for i in range(3)], 
                                        name=coord_sys)
        function_range = CoordinateSystem(['x','y','z'], name='world')
        spatial_coordmap = AffineTransform(function_domain, function_range,
                                           affine)

        nonspatial_names = ['axis%d' % i for i in range(3, data.ndim)]
        if nonspatial_names:
            nonspatial_coordmap = AffineTransform.from_start_step(nonspatial_names, nonspatial_names, [0]*(data.ndim-3), [1]*(data.ndim-3))
            full_coordmap = cmap_product(spatial_coordmap, nonspatial_coordmap)
        else:
            full_coordmap = spatial_coordmap

        self._spatial_coordmap = spatial_coordmap

        self.coord_sys = coord_sys
        Image.__init__(self, data, full_coordmap) 
        if metadata is not None:
            self.metadata = metadata

    def _get_spatial_coordmap(self):
        """
        Returns 3 dimensional AffineTransform, which is the same
        as self.coordmap if self.ndim == 3. 
        """
        return self._spatial_coordmap
    spatial_coordmap = property(_get_spatial_coordmap)

    def _get_affine(self):
        """
        Returns the affine of the spatial coordmap which will
        always be a 4x4 matrix.
        """
        return self._spatial_coordmap.affine
    affine = property(_get_affine)

    def get_data(self):
        # XXX What's wrong with __array__? Wouldn't that be closer to numpy?
        """ Return data as a numpy array.
        """
        return np.asarray(self._data)

    def resampled_to_affine(self, affine_transform, world_to_world=None, 
                            interpolation_order=3, 
                            shape=None):
        """ Resample the image to be an affine image.

            Parameters
            ----------
            affine_transform : AffineTransform

                Affine of the new grid. 

                XXX In the original proposal, it said something about "if only 3x3 it is assumed
                to be a rotation", but this wouldn't work the way the code was written becuase
                it was written as if affine was the affine of an AffineImage. So, if you input
                a "rotation matrix" that is assuming you have voxels of size 1....
                This rotation can now be expressed with the world_to_world argument.

            world_to_world: 4x4 ndarray, optional
                A matrix representing a mapping from the target's (affine_transform) "world"
                to self's "world". Defaults to np.identity(4)

            interpolation_order : int, optional
                Order of the spline interplation. If 0, nearest-neighbour
                interpolation is performed.

            shape: tuple
                Shape of the resulting image. Defaults to self.shape.

            Returns
            -------
            resampled_image : nipy AffineImage
                New nipy image with the data resampled in the given
                affine.

            Notes
            -----
            The coordinate system of the resampled_image is the world
            of affine_transform. Therefore, if world_to_world=np.identity(4),
            the coordinate system is not changed: the
            returned image points to the same world space.

        """

        shape = shape or self.shape
        shape = shape[:3]

        if world_to_world is None:
            world_to_world = np.identity(4)
        world_to_world_transform = AffineTransform(affine_transform.function_range,
                                                   self.spatial_coordmap.function_range,
                                                   world_to_world)
        # Delayed import to avoid circular imports
        from ...algorithms.resample import resample
        if self.ndim == 3:
            im = resample(self, affine_transform, world_to_world_transform,
                          shape, order=interpolation_order)
            return AffineImage(np.array(im), affine_transform.affine,
                               affine_transform.function_domain.name)

        # XXX this below wasn't included in the original AffineImage proposal
        # and it would fail for an AffineImage with ndim == 4.
        # I don't know if it should be included as a special case in the AffineImage,
        # but then we should at least raise an exception saying that these resample_* methods
        # only work for AffineImage's with ndim==3.
        #
        # This is part of the reason nipy.core.image.Image does not have
        # resample_* methods...

        elif self.ndim == 4:

            result = np.empty(shape + (self.shape[3],))
            data = self.get_data()
            for i in range(self.shape[3]):
                tmp_affine_im = AffineImage(data[...,i], self.affine,
                                            self.axis_names[:-1])
                tmp_im = tmp_affine_im.resampled_to_affine(affine_transform, 
                                                           world_to_world,
                                                           interpolation_order,
                                                           shape)

                result[...,i] = np.array(tmp_im)
            return AffineImage(result, affine_transform.affine,
                               affine_transform.function_domain.name)
        else:
            raise ValueError('resampling only defined for 3d and 4d AffineImage')


    def resampled_to_img(self, target_image, world_to_world=None, interpolation_order=3):
        """ Resample the image to be on the same grid than the target image.

            Parameters
            ----------
            target_image : AffineImage
                Nipy image onto the grid of which the data will be
                resampled.
            XXX In the proposal, target_image was assumed to be a matrix if it had no attribute "affine". It now has to have a spatial_coordmap attribute.
            
            world_to_world: 4x4 ndarray, optional
                A matrix representing a mapping from the target's "world"
                to self's "world". Defaults to np.identity(4)


            interpolation_order : int, optional
                Order of the spline interplation. If 0, nearest neighboor 
                interpolation is performed.

            Returns
            -------
            resampled_image : nipy_image
                New nipy image with the data resampled.

            Notes
            -----
            The coordinate system of the resampled_image is the world
            of target_image. Therefore, if world_to_world=np.identity(4),
            the coordinate system is not changed: the
            returned image points to the same world space.


XXX Since you've enforced the outputs always to be 'x','y','z' -- EVERY image is embedded in the same coordinate system (i.e. 'x','y','z'), but images can have different coordinate axes. The term "embedding" that was here in the proposal refers to something in the range of a function, not its domain. By adding a world_to_world transformation, i.e. a rotation or something, we
now change the coordinate system of the resampled_image

        """
        return self.resampled_to_affine(target_image.spatial_coordmap,
                                        world_to_world,
                                        interpolation_order,
                                        target_image.shape)


    def values_in_world(self, x, y, z, interpolation_order=3):
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
            interpolation_order : int, optional
                Order of the spline interplation. If 0, nearest neighboor 
                interpolation is performed.

            Returns
            -------
            values : number or ndarray
                Data values interpolated at the given world position.
                This is a number or an ndarray, depending on the shape of
                the input coordinate.
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)
        shape = x.shape
        if not ((x.shape == y.shape) and (x.shape == z.shape)):
            raise ValueError('x, y and z shapes should be equal')
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()
        xyz = np.c_[x, y, z]
        world_to_voxel = self.spatial_coordmap.inverse()
        ijk = world_to_voxel(xyz)
        values = ndimage.map_coordinates(self.get_data(), ijk.T,
                                    order=interpolation_order)
        values = np.reshape(values, shape)
        return values
    
    #---------------------------------------------------------------------------
    # AffineImage interface
    #---------------------------------------------------------------------------

    def xyz_ordered(self):
        """ Returns an image with the affine diagonal and positive
            in its coordinate system.
        """
        A, b = to_matrix_vector(self.affine)
        if not np.all((np.abs(A) > 0.001).sum(axis=0) == 1):
            raise CoordSystemError(
                'Cannot reorder the axis: the image affine contains rotations'
                )
        axis_numbers = list(np.argmax(np.abs(A), axis=1))
        axis_names = [self.spatial_coordmap.function_domain.coord_names[a] for a in axis_numbers]
        reordered_coordmap = self.spatial_coordmap.reordered_domain(axis_names)
        data = self.get_data()
        transposed_data = np.transpose(data, axis_numbers + range(3, self.ndim))
        return AffineImage(transposed_data, reordered_coordmap.affine,
                           reordered_coordmap.function_domain.name)
    
    #---------------------------------------------------------------------------
    # Private methods
    #---------------------------------------------------------------------------
    
    def __repr__(self):
        options = np.get_printoptions()
        np.set_printoptions(precision=6, threshold=64, edgeitems=2)
        representation = \
                'AffineImage(\n  data=%s,\n  affine=%s,\n  coord_sys=%s)' % (
                '\n       '.join(repr(self._data).split('\n')),
                '\n         '.join(repr(self.affine).split('\n')),
                repr(self.coord_sys))
        np.set_printoptions(**options)
        return representation


    def __copy__(self):
        """ Copy the Image and the arrays and metadata it contains.
        """
        return self.__class__(data=self.get_data().copy(), 
                              affine=self.affine.copy(),
                              coord_sys=self.coord_sys,
                              metadata=self.metadata.copy())


    def __deepcopy__(self, option):
        """ Copy the Image and the arrays and metadata it contains.
        """
        import copy
        return self.__class__(data=self.get_data().copy(), 
                              affine=self.affine.copy(),
                              coord_sys=self.coord_sys,
                              metadata=copy.deepcopy(self.metadata))


    def __eq__(self, other):
        return (    isinstance(other, self.__class__)
                and np.all(self.get_data() == other.get_data())
                and np.all(self.affine == other.affine)
                and (self.coord_sys == other.coord_sys))


