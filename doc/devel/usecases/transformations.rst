.. _transformations:

==========================
 Transformation use cases
==========================

Use cases for defining and using transforms on images.

I have an image *Img*.  

Image Orientation
-----------------

I would like to know what the voxel sizes are
I would like to determine whether it was acquired axially,
coronally or sagittally.  What is the brain orientation in relation to
the voxels?  Has it been acquired at an oblique angle?  What are the
voxel dimensions?::

  img = load_image(file)
  cm = img.coordmap
  print cm
  
  input_coords axis0: Inferior -> Superior
	       axis1: Posterior -> Anterior
	       axis2: Right -> Left
 	       
	       effective pixel dimensions
			      axis0: 4mm
			      axis1: 2mm
			      axis2: 2mm

  input/output mapping
		 <Affine Matrix>

		 input axis0 maps exactly to output axis2
		 input axis1 maps exactly to output axis1
		 input axis2 maps exactly to output axis0

  output_coords axis0: Left -> Right
		axis1: Posterior -> Anterior
		axis2: Inferior -> Superior


In the case of a mapping that does not exactly align the input and
output axes, something like::

  ...
  input/output mapping
		 <Affine Matrix>

		 input axis0 maps closest to output axis2
		 input axis1 maps closest to output axis1
		 input axis2 maps closest to output axis0
  ...

Creating transformations / co-ordinate maps
-------------------------------------------

I have an array *pixelarray* that represents voxels in an image and have a
matrix/transform *mat* which represents the relation between the voxel
coordinates and the coordinates in scanner space (world coordinates).
I want to associate the array with the matrix::

  img = load_image(infile)
  pixelarray = np.asarray(img)

(*pixelarray* is an array and does not have a coordinate map.)::

  pixelarray.shape
  (40,256,256)

So, now I have some arbitrary transformation matrix::

  mat = np.zeros((4,4))
  mat[0,2] = 2 # giving x mm scaling
  mat[1,1] = 2 # giving y mm scaling
  mat[2,0] = 4 # giving z mm scaling
  mat[3,3] = 1 # because it must be so
  # Note inverse diagonal for zyx->xyz coordinate flip
  
I want to make an ``Image`` with these two::

  coordmap = voxel2mm(pixelarray.shape, mat)
  img = Image(pixelarray, coordmap)

The ``voxel2mm`` function allows separation of the image *array* from
the size of the array, e.g.::

  coordmap = voxel2mm((40,256,256), mat)

We could have another way of constructing image which allows passing
of *mat* directly::

  img = Image(pixelarray, mat=mat)  

or::

  img = Image.from_data_and_mat(pixelarray, mat)

but there should be "only one (obvious) way to do it".

I have two images, ImageA and ImageB.  Each image has a voxel-to-world
transform associated with it.  (The "world" for these two transforms
could be similar or even identical in the case of an fmri series.)  I
would like to get from voxel coordinates in ImageA to voxel
coordinates in ImageB, for resampling::

  imgA = load_image(infile_A)
  cmA = imgA.coordmap
  imgB = load_image(infile_B)
  cmB = imgB.coordmap
  # I want to first apply transform implied in 
  # cmA, then the inverse of transform implied in 
  # cmB.  If these are matrices then this would be
  # np.dot(cm.inverse(cmB), cmA)
  voxA_to_voxB = cm.compose(cm.inverse(cmB), cmA)

(rather than this, on the basis that people need to understand the
mathematics of function composition to some degree)::

  voxA_to_voxB = cm.firsta_thenb(cmA, cm.inverse(cmB))

see wikipedia_function_composition_.

.. _wikipedia_function_composition: http://en.wikipedia.org/wiki/Function_composition

I have done a coregistration between two images, ImageA and ImageB.
This has given me a voxel-to-voxel transformation and I want to store
this transformation in such a way that I can use this transform to
resample ImageA to ImageB.  :ref:`resampling`

I have done a coregistration between two images, ImageA and ImageB. I
may want this to give me a worldA-to-worldB transformation, where
worldA is the world of voxel-to-world for ImageA, and worldB is the
world of voxel-to-world of ImageB.  

My *ImageA* has a voxel to world transformation.  This transformation
may (for example) have come from the scanner that acquired the image -
so telling me how the voxel positions in *ImageA* correspond to
physical coordinates in terms of the magnet isocenter and millimeters
in terms of the primary gradient orientations (x, y and z). I have the
same for *ImageB*.  For example, I might choose to display this image
resampled so each voxel is a 1mm cube.

Now I have these transformations:  ST(ImageA-V2W), and
ST(ImageB-V2W) (where ST is *scanner tranform* as above, and *V2W* is
voxel to world).

I have now done a coregistration between *ImageA* and *ImageB*
(somehow) - giving me, in addition to *ImageA* and *ImageB*, a
transformation that registers *ImageA* and *ImageB*. Let's call this
tranformation V2V(ImageA, ImageB), where V2V is voxel-to-voxel.

In actuality ImageB can be an array of images, such as series of fMRI
images and I want to align all the ImageB series to ImageA and then
take these voxel-to-voxel aligned images (the ImageA and ImageB array)
and remap them to the world space (voxel-to-world). Since remapping is
an interpolation operation I can generate errors in the resampled
pixel values. If I do more than one resampling, error will
accumulate. I want to do only a single resampling. To avoid the errors
associated with resampling I will build a *composite transformation*
that will chain the separate voxel-to-voxel and voxel-to-world
transformations into a single transformation function (such as an
affine matrix that is the result of multiplying the several affine
matrices together). With this single *composite transformatio* I now
resample ImageA and ImageB and put them into the world coordinate
system from which I can make measurements.

