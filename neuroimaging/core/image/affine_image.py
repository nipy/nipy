"""
The base image interface.
"""

import numpy as np
from scipy import ndimage

# Local imports
from neuroimaging.core.transforms.affines import from_matrix_vector, \
                     to_matrix_vector
from neuroimaging.core.api import Affine as AffineTransform

from base_image import CoordSystemError, BaseImage

################################################################################
# class `AffineImage`
################################################################################

class AffineImage(BaseImage):
    """ The affine image for neuroimaging.

        This object is nothing more than an ndarray representing a
        volume, with the first 3 dimensions being spatial, and mapped to 
        a named coordinate system using an affine (4x4 matrix).

        **Attributes**

        :affine: 4x4 ndarray

            Affine mapping from indices to world coordinates.

        :coord_sys: string or coord_sys object

            Coordinate system the data is embedded in. For
            instance `neuroimaging.refs.mni152`.

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
    coord_sys = ''

    # User defined meta data
    metadata = dict()

    # The data (ndarray)
    _data = None

    #---------------------------------------------------------------------------
    # Attributes, AffineImage interface
    #---------------------------------------------------------------------------

    # The affine (4x4 ndarray)
    affine = np.eye(4)

    #---------------------------------------------------------------------------
    # BaseImage interface
    #---------------------------------------------------------------------------

    def __init__(self, data, affine, coord_sys, metadata=None):
        """ Creates a new neuroimaging image with an affine mapping.

            Parameters
            ----------

            data : ndarray
                ndarray representing the data.
            affine : 4x4 ndarray
                affine transformation to the reference coordinate system
            coord_system : string
                name of the reference coordinate system.
        """
        self._data = data
        if not affine.shape == (4, 4):
            raise ValueError('The affine should be a 4x4 array')
        self.affine = affine
        self.coord_sys = coord_sys
        if metadata is not None:
            self.metadata = metadata

    
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
            transform : neuroimaging.core.Transform object
        """
        # XXX: Affine.from_params does not have this signature.
        return AffineTransform.from_params(self.affine, self.coord_sys)


    def resampled_to_affine(self, affine=None, interpolation_order=3):
        """ Resample the image to be an affine image.

            Parameters
            ----------
            affine : 4x4 ndarray or 3x3 ndarray
                Affine of the new grid or transform object pointing
                to the new grid. If a 3x3 ndarray is given, it is
                considered to be the rotation part of the affine, and the 
                best possible bounding box is calculated.
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
            The coordinate system of the image is not changed: the
            returned image points to the same world space.
        """
        if affine is None:
            return self
        transform_affine = np.dot(np.linalg.inv(self.affine), affine)
        if transform_affine.shape == (3, 3):
            A = transform_affine
            # XXX: implement the algorithm to find out optimal b, 
            # And we need to modify affine, as it is a 3x3 array, not 4x4
            raise NotImplementedError
            b = None
            transform_affine = from_matrix_vector(A, b)
        else:
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
        return AffineImage(resampled_data, affine, 
                           self.coord_sys, metadata=self.metadata)


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
            embedded in the same coordinate system.
        """
        if not target_image.coord_sys == self.coord_sys:
            raise CoordSystemError(
                'The two images do not point to the same coordinate system')
        target_shape = target_image.get_data().shape[:3]
        if hasattr(target_image, 'affine'):
            return self.resampled_to_grid(target_image.affine,
                                    target_shape,
                                    interpolation_order=interpolation_order)
        else:
            # XXX: we need a dispatcher pattern or to encode the
            # information in the transform
            return super(AffineImage, self).resampled_to_img(target_image,
                                    interpolation_order=interpolation_order)


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
        xyz = np.c_[x, y, z, np.ones_like(x)]
        ijk = np.dot(np.linalg.inv(self.affine), xyz.T)[:3]
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
        return AffineImage(reordered_data, new_affine, self.coord_sys, 
                                           metadata=self.metadata)

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


