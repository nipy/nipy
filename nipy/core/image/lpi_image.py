
"""
The base image interface.
"""

import numpy as np
from scipy.ndimage import map_coordinates

# Local imports
from nipy.core.transforms.affines import from_matrix_vector, \
                     to_matrix_vector
from nipy.core.api import AffineTransform, Image, CoordinateSystem
from nipy.core.reference.coordinate_map import compose, product as cmap_product
from nipy.algorithms.resample import resample


#  Name of dimensions are based on 
# 'x+LR' = x increasing from patient's L to R
# 'y+PA' = x increasing from patient's P to A
# 'z+SI' = x increasing from patient's S to I


lps_output_coordnames = ('x+LR', 'y+PA', 'z+SI') 
ras_output_coordnames = ('x+RL', 'y+AP', 'z+SI') 

################################################################################
# class `LPITransform`
################################################################################

class LPITransform(AffineTransform):


   range = CoordinateSystem('xyz', name='world-LPI')

   def __init__(self, affine, lpi_axis_names):

      """
      >>> lpi = LPITransform(np.diag([3,4,5,1]), 'ijk')
      >>> lpi
      LPITransform(
         function_domain=CoordinateSystem(coord_names=('i', 'j', 'k'), name='voxel', coord_dtype=float64),
         function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='world-LPI', coord_dtype=float64),
         affine=array([[ 3.,  0.,  0.,  0.],
                       [ 0.,  4.,  0.,  0.],
                       [ 0.,  0.,  5.,  0.],
                       [ 0.,  0.,  0.,  1.]])
      )
      >>> 
      """

      if affine.shape != (4,4):
          raise ValueError('affine must be a 4x4 matrix representing an affine transformation in homogeneous coordinates')
      domain = CoordinateSystem(lpi_axis_names, name='voxel')
      range = CoordinateSystem('xyz', name='world')
      AffineTransform.__init__(self, domain, self.range, affine)

   def reordered_range(self, order, name=''):
       raise NotImplementedError("the LPI world coordinates are always ['x','y','z'] and can't be reordered")

   def renamed_range(self, newnames, name=''):
       raise NotImplementedError("the LPI world coordinates are always ['x','y','z'] and can't be renamed")

   def __repr__(self):
       s_split = AffineTransform.__repr__(self).split('\n')
       s_split[0] = 'LPITransform('
       return '\n'.join(s_split)
       
################################################################################
# class `LPIImage`
################################################################################

class LPIImage(Image):

   """ The standard image for nipy, an Image with
       LPI output coordinates.

       This object is a subclass of Image that
       assumes the first 3 coordinates
       are spatial. 

       **Attributes**

       :metadata: dictionary

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

       :lpi_transform: LPITransform

           A CoordinateMap that relates all the spatial axes of the data
           to LPI 'xyz' coordinates.

       :coordmap: AffineTransform

           Coordinate map describing the relationship between
           all coordinates and axis_names.

       :axes: CoordinateSystem
  
           CoordinateSystem for the axes of the data.

       :world: CoordinateSystem
 
           CoordinateSystem for the world space of the data.


   Notes
   -----

   The data is stored in an undefined way: prescalings might need to
   be applied to it before using it, or the data might be loaded on
   demand. The best practice to access the data is not to access the
   _data attribute, but to use the `get_data` method.

   >>> data = np.empty((30,40,50))
   >>> affine = np.diag([3,4,5,1])
   >>> im = LPIImage(data, affine, 'ijk')
   >>> im.world
   CoordinateSystem(coord_names=('x', 'y', 'z'), name='world-LPI', coord_dtype=float64)
   >>> im.axes
   CoordinateSystem(coord_names=('i', 'j', 'k'), name='voxel', coord_dtype=float64)
   >>> im.lpi_transform
   LPITransform(
      function_domain=CoordinateSystem(coord_names=('i', 'j', 'k'), name='voxel', coord_dtype=float64),
      function_range=CoordinateSystem(coord_names=('x', 'y', 'z'), name='world-LPI', coord_dtype=float64),
      affine=array([[ 3.,  0.,  0.,  0.],
                    [ 0.,  4.,  0.,  0.],
                    [ 0.,  0.,  5.,  0.],
                    [ 0.,  0.,  0.,  1.]])
   )
   >>> 

   """



   #---------------------------------------------------------------------------
   # Attributes
   #---------------------------------------------------------------------------
   
   # User defined meta data
   metadata = dict()

   # The data (ndarray)
   _data = None

   # XXX: Need an attribute to determine in a clever way the
   # interplation order/method

   def __init__(self, data, affine, axis_names, metadata={}):
      """ Creates a new nipy image with an affine mapping.

      Parameters
      ----------

      data : ndarray
         ndarray representing the data.

      affine : 4x4 ndarray
         affine transformation to the reference coordinate system

      axis_names : [string]
         names of the axes in the coordinate system.
      """

      if len(axis_names) < 3:
         raise ValueError('LPIImage must have a minimum of 3 axes')

      # The first three axes are assumed to be the
      # spatial ones
      lpi_transform = LPITransform(affine, axis_names[:3])
      nonspatial_names = axis_names[3:]
        
      if nonspatial_names:
         nonspatial_affine_transform = AffineTransform.from_start_step(nonspatial_names, nonspatial_names, [0]*(data.ndim-3), [1]*(data.ndim-3))
         full_dimensional_affine_transform = cmap_product(lpi_transform, nonspatial_affine_transform)
      else:
         full_dimensional_affine_transform = lpi_transform 

      self._lpi_transform = lpi_transform

      Image.__init__(self, data, full_dimensional_affine_transform,
                     metadata=metadata)

   #---------------------------------------------------------------------------
   # Overwriting some parts of the Image interface
   #---------------------------------------------------------------------------

   def _getworld(self):
      return self.lpi_transform.function_range # == LPITransform.range
   world = property(_getworld, doc="World space.")

   def _get_affine(self):
      """
      Returns the affine of the LPITransform which will
      always be a 4x4 matrix.
      """
      return self.lpi_transform.affine
   affine = property(_get_affine, doc="4x4 Affine matrix")

   def reordered_world(self, order):
      raise NotImplementedError("the LPI world coordinates are always ['x','y','z'] and can't be reordered")

   def reordered_axes(self, order=None):

      """
      Return a new LPIImage whose axes have been reordered.
      The reordering must be such that the first 3
      coordinates remain the same.

      Parameters
      ----------

      order : sequence
          Order to use, defaults to reverse. The elements
          can be integers, strings or 2-tuples of strings.
          If they are strings, they should be in 
          self.axes.coord_names.

      name: string, optional
          Name of new function_domain, defaults to self.function_domain.name.

      Returns:
      --------

      im_reordered: LPIImage

      Examples:
      ---------

      >>> im = LPIImage(np.empty((30,40,50)), np.diag([2,3,4,1]), 'ijk')
      >>> im_reordered = im.reordered_axes([2,0,1])
      >>> im_reordered.shape
      (50, 30, 40)
      >>> im_reordered.affine
      array([[ 0.,  2.,  0.,  0.],
             [ 0.,  0.,  3.,  0.],
             [ 4.,  0.,  0.,  0.],
             [ 0.,  0.,  0.,  1.]])

      >>> im_reordered2 = im.reordered_axes('kij')
      >>> im_reordered2.shape
      (50, 30, 40)
      >>> im_reordered2.affine
      array([[ 0.,  2.,  0.,  0.],
             [ 0.,  0.,  3.,  0.],
             [ 4.,  0.,  0.,  0.],
             [ 0.,  0.,  0.,  1.]])
      >>> 

      """

      if order is None:
         order = range(self.ndim)[::-1]
      elif type(order[0]) == type(''):
         order = [self.axes.index(s) for s in order]

      if set(order[:3]) != set(range(3)):
         raise ValueError('the reordering must keep the first three axes unchanged')

      # Reordering the input
      # will transpose the data, so we will have accessed the data.

      im = self.to_image().reordered_axes(order)
      
      A = np.identity(4)
      A[:3,:3] = im.affine[:3,:3]
      A[:3,-1] = im.affine[:3,-1]

      return LPIImage(np.array(im), A, im.axes.coord_names,
                      metadata=self.metadata)

   def renamed_world(self, newnames, name=''):
       raise NotImplementedError("the LPI world coordinates are always ['x','y','z'] and can't be renamed")

   def renamed_axes(self, **names_dict):
      """
      Return a new image with its axes renamed according
      to the dictionary.

      Parameters
      ----------

      img : Image
      
      names_dict : dictionary

      Returns
      -------

      newimg : Image
         An Image with the same data, having its axes renamed.
      
      Examples
      --------

      >>> data = np.random.standard_normal((11,9,4))
      >>> im = Image(data, AffineTransform.from_params('ijk', 'xyz', np.identity(4)))
      >>> im_renamed = im.renamed_axes(i='slice')
      >>> print im_renamed.axes
      CoordinateSystem(coord_names=('slice', 'j', 'k'), name='domain', coord_dtype=float64)

      """

      newim = self.to_image().renamed_axes(**names_dict)
      return LPIImage.from_image(newim)


   #---------------------------------------------------------------------------
   # LPIImage interface
   #---------------------------------------------------------------------------


   #---------------------------------------------------------------------------
   # Properties
   #---------------------------------------------------------------------------

   def _get_lpi_transform(self):
      """
      Returns 3-dimensional LPITransform, which is the same
      as self.coordmap if self.ndim == 3. 
      """
      return self._lpi_transform
   lpi_transform = property(_get_lpi_transform)


   #---------------------------------------------------------------------------
   # Methods
   #---------------------------------------------------------------------------

   def to_image(self):
      """
      Return an Image with the same data as self.
      """
      import copy
      return Image(self._data, copy.copy(self.coordmap))

   @staticmethod
   def from_image(img):
      """
      Return an LPIImage from an Image with the same data.
      The affine matrix is read off from the upper left corner
      of img.affine.
      """
      A = np.identity(4)
      A[:3,:3] = img.affine[:3,:3]
      A[:3,-1] = img.affine[:3,-1]
      return LPIImage(img._data, A, img.axes.coord_names)

   def resampled_to_affine(self, affine_transform, world_to_world=None, interpolation_order=3, 
                           shape=None):
      """ Resample the image to be an affine image.

      Parameters
      ----------
      affine_transform : LPITransform
         Affine of the new grid. 

      world_to_world: 4x4 ndarray, optional
         A matrix representing a mapping from the target's "world"
         to self's "world". Defaults to np.identity(4)

      interpolation_order : int, optional
         Order of the spline interplation. If 0, nearest-neighbour
         interpolation is performed.

      shape: tuple
         Shape of the resulting image. Defaults to self.shape.

      Returns
      -------

      resampled_image : LPIImage
         New nipy image with the data resampled in the given
         affine.

      Notes
      -----

      The coordinate system of the output image is the world
      of affine_transform. Therefore, if world_to_world=np.identity(4),
      the coordinate system is not changed: the
      returned image points to the same world space.


      """

      shape = shape or self.shape
      shape = shape[:3]

      if world_to_world is None:
         world_to_world = np.identity(4)
      world_to_world_transform = AffineTransform(affine_transform.function_range,
                                                 self.world, world_to_world)

      if self.ndim == 3:
         im = resample(self, affine_transform, world_to_world_transform,
                       shape, order=interpolation_order)
         return LPIImage(np.array(im), affine_transform.affine,
                         affine_transform.function_domain.coord_names,
                         metadata=self.metadata)

        # XXX this below wasn't included in the original LPIImage proposal
        # and it would fail for an LPIImage with ndim == 4.
        # I don't know if it should be included as a special case in the LPIImage,
        # but then we should at least raise an exception saying that these resample_* methods
        # only work for LPIImage's with ndim==3.
        #
        # This is part of the reason nipy.core.image.Image does not have
        # resample_* methods...

      elif self.ndim == 4:
         
         result = np.empty(shape + (self.shape[3],))
         data = self.get_data()
         for i in range(self.shape[3]):
            tmp_affine_im = LPIImage(data[...,i], self.affine,
                                     self.axes.coord_names[:-1],
                                     metadata=self.metadata)
            tmp_im = tmp_affine_im.resampled_to_affine(affine_transform, 
                                                       world_to_world,
                                                       interpolation_order,
                                                       shape)

            result[...,i] = np.array(tmp_im)
         return LPIImage(result, affine_transform.affine,
                         self.axes.coord_names,
                         metadata=self.metadata)
      else:
         raise ValueError('resampling only defined for 3d and 4d LPIImage')

   def resampled_to_img(self, target_image, world_to_world=None, interpolation_order=3):
      """ Resample the image to be on the same grid than the target image.
      
      Parameters
      ----------
      target_image : LPIImage
         LPIImage onto the grid of which the data will be
         resampled.

      world_to_world: 4x4 ndarray, optional
         A matrix representing a mapping from the target's "world"
         to self's "world". Defaults to np.identity(4)

      interpolation_order : int, optional
         Order of the spline interplation. If 0, nearest neighboor 
         interpolation is performed.

      Returns
      -------
      resampled_image : LPIImage
         New LPIImage with the data resampled.

      """
      return self.resampled_to_affine(target_image.lpi_transform,
                                      interpolation_order=interpolation_order,
                                      shape=target_image.shape,
                                      world_to_world=world_to_world)

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
      world_to_voxel = self.lpi_transform.inverse()
      ijk = world_to_voxel(xyz)

      data = self.get_data()

      if self.ndim == 3:
         values = map_coordinates(data, ijk.T,
                                  order=interpolation_order)
         values = np.reshape(values, shape)
      elif self.ndim == 4:
         values = np.empty(shape + (self.shape[3],))
         for i in range(self.shape[3]):
            tmp_values = map_coordinates(data[...,i], ijk.T,
                                         order=interpolation_order)
            tmp_values = np.reshape(tmp_values, shape)
            values[...,i] = tmp_values
      return values
    
   def xyz_ordered(self, positive=False):
      """ 
      Returns an image with the affine diagonal, (optionally
      with positive entries),
      in the LPI coordinate system.

      Parameters
      ----------
      
      positive : bool, optional
         If True, also ensures that the diagonal entries are positive.

      Notes
      -----
      This may possibly transpose the data array.

      If positive is True, this may involve
      creating a new array with data
      
      self.get_data()[::-1,::-1]

      """
      A, b = to_matrix_vector(self.affine)
      if not np.all((np.abs(A) > 0.001).sum(axis=0) == 1):
         raise ValueError(
            'Cannot reorder the axis: the image affine contains rotations'
            )
      axis_numbers = list(np.argmax(np.abs(A), axis=1))
      im = self.reordered_axes(axis_numbers + range(3, self.ndim))

      if not positive:
         return im
      else:
         # Determine which axes, if any, have to be flipped in the array
         slice_list = []
         diag_values = np.diag(im.affine)
         for value in np.diag(im.affine)[:-1]:
            if value < 0:
               slice_list.append(slice(None,None,-1))
            else:
               slice_list.append(slice(None,None,None))

         if slice_list == [slice(None,None,None)]*3:
            # do nothing
            return im
         else:
            from nipy.core.image.image import subsample
            im = subsample(im.to_image(), tuple(slice_list))
            return LPIImage.from_image(im)

    #---------------------------------------------------------------------------
    # Private methods
    #---------------------------------------------------------------------------
    
   def __repr__(self):
      options = np.get_printoptions()
      np.set_printoptions(precision=6, threshold=64, edgeitems=2)
      representation = \
          'LPIImage(\n  data=%s,\n  affine=%s,\n  axis_names=%s)' % (
         '\n       '.join(repr(self._data).split('\n')),
         '\n         '.join(repr(self.affine).split('\n')),
         repr(self.axes.coord_names))
      np.set_printoptions(**options)
      return representation


   def __copy__(self):
      """ Copy the Image and the arrays and metadata it contains.
        """
      return self.__class__(data=self.get_data().copy(), 
                            affine=self.affine.copy(),
                            axis_names=self.axes.coord_names,
                            metadata=self.metadata.copy())


   def __deepcopy__(self, option):
      """ Copy the Image and the arrays and metadata it contains.
      """
      import copy
      return self.__class__(data=self.get_data().copy(), 
                            affine=self.affine.copy(),
                            axis_names=self.axes.coord_names,
                            metadata=copy.deepcopy(self.metadata))



