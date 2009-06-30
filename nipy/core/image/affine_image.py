"""
The base image interface.
"""

import numpy as np

# Local imports
from nipy.core.transforms.affines import from_matrix_vector, \
                     to_matrix_vector
from nipy.core.api import Affine as AffineTransform, Image, CoordinateSystem
from nipy.core.reference.coordinate_map import compose, product as cmap_product, reorder_input
from nipy.algorithms.resample import resample

################################################################################
# class `AffineImage`
################################################################################

class AffineImage(Image):

    """ The affine image for nipy.

        This object is a subclass of Image that
        assumes the first 3 coordinates
        are spatial. 

        **Attributes**

        :affine: 4x4 ndarray

            Affine mapping from voxel axes to world coordinates
            (world coordinates are always forced to be 'x', 'y', 'z').

        :coordmap: AffineTransform

            Coordinate map describing the spatial coordinates 
            (always forced to be 'x', 'y', 'z') and the coordinate
            axes with names axis_names.
           
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

    # The name of the reference coordinate system
    axis_names = ['i', 'j', 'k']

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
        input_coords = CoordinateSystem(['axis%d' % i for i in range(3)], 
                                        name=coord_sys)
        output_coords = CoordinateSystem(['x','y','z'], name='world')
        coordmap = AffineTransform(affine, input_coords, output_coords)

        nonspatial_names = ['axis%d' % i for i in range(3, data.ndim)]
        if nonspatial_names:
            nonspatial_coordmap = AffineTransform.from_start_step(nonspatial_names, nonspatial_names, [0]*(data.ndim-3), [1]*(data.ndim-3))
            full_coordmap = cmap_product(coordmap, nonspatial_coordmap)
        else:
            full_coordmap = spatial_coordmap 
        self._3dcoordmap = coordmap
        Image.__init__(self, data, full_coordmap) # XXX currently, Image would have to not
                                 # check the dimensions of coordmap...
        if metadata is not None:
            self.metadata = metadata

    def get_3dcoordmap(self):
        """
        Returns 3 dimensional AffineTransform, which is the same
        as self.coordmap if self.ndim == 3. 
        """
        return self._3dcoordmap

    def get_data(self):
        """ Return data as a numpy array.
        """
        return np.asarray(self._data)

    def get_transform(self):
        """ Returns the transform object associated with the image which is a 
            general description of the mapping from the voxel grid to the 
            world space.

            Returns
            -------
            transform : nipy.core.Transform object
        """
        return self.coordmap

    def resampled_to_affine(self, affine=None, interpolation_order=3, 
                            shape=None):
        """ Resample the image to be an affine image.

            Parameters
            ----------
            affine : AffineTransform

                Affine of the new grid or transform object pointing
                to the new grid. If a 3x3 ndarray is given, it is
                considered to be the rotation part of the affine, and the 
                best possible bounding box is calculated.

            interpolation_order : int, optional
                Order of the spline interplation. If 0, nearest-neighboor 
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
            The coordinate system of the image is not changed: the
            returned image points to the same world space.

        """

        shape = shape or self.shape
        target = affine
        target_world_to_self_world = compose(self.coordmap,
                                             target.inverse)
        return resample(self, target, target_world_to_self_world,
                        self.shape, interpolation_order)

    def resampled_to_img(self, target_image, interpolation_order=3):
        """ Resample the image to be on the same grid than the target image.

            Parameters
            ----------
            target_image : nipy image
                Nipy image onto the grid of which the data will be
                resampled.
            XXX In the proposal, target_image was assumed to be a matrix if it had no attribute "affine". It now has to have a coordmap attribute.
            
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
            embedded in the same coordinate system.

XXX Since you've enforced the outputs always to be 'x','y','z' -- EVERY image is embedded in the same coordinate system (i.e. 'x','y','z'), but images can have different coordinate axes. Here it should say that the coordinate axes are the same. The term "embedding" refers to something in the range of a function, not its domain. 

        """
        return self.resampled_to_affine(target_image.coordmap,
                                        interpolation_order=interpolation_order,
                                        shape=target.shape)

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
        world_to_voxel = self.coordmap.inverse
        ijk = world_to_voxel(xyz)
        values = ndimage.map_coordinates(self.get_data(), ijk,
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
        img = self
        axis_numbers = np.argmax(np.abs(A), axis=0)
        axis_names = [self.coordmap.input_coords.coord_names[a] for a in axis_numbers]
        reordered_coordmap = reorder_input(self.coordmap, axis_names)
        data = self.get_data()
        transposed_data = np.transpose(data, axis_numbers)
        return AffineImage(transposed_data, reordered_coordmap.affine,
                           reordered_coordmap.input_coords.name)
    
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


