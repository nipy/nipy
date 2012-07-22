.. _transformations:

==========================
 Transformation use cases
==========================

Use cases for defining and using transforms on images.

We should be very careful to only use the terms ``x, y, z`` to refer to
physical space.  For voxels, we should use ``i, j, k``, or ``i', j', k'`` (i
prime, j prime k prime).


I have an image *Img*.  

Image Orientation
-----------------

I would like to know what the voxel sizes are.

I would like to determine whether it was acquired axially,
coronally or sagittally.  What is the brain orientation in relation to
the voxels?  Has it been acquired at an oblique angle?  What are the
voxel dimensions?::

  img = load_image(file)
  cm = img.coordmap
  print cm
  
  input_coords axis_i:
	       axis_j: 
	       axis_k: 
 	       
	       effective pixel dimensions
			      axis_i: 4mm
			      axis_j: 2mm
			      axis_k: 2mm

  input/output mapping
		 <Affine Matrix>



		 
		     x   y   z                    
                   ------------
                 i|  90  90   0
		 j|  90   0  90
		 k| 180	 90  90	 

		 input axis_i maps exactly to output axis_z
		 input axis_j maps exactly to output axis_y
		 input axis_k maps exactly to output axis_x flipped 180

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


If the best matching axis is reversed compared to input axis::

  ...
  input axis0 maps [closest|exactly] to negative output axis2 

and so on.

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

Composing transforms
''''''''''''''''''''

I have two images, *img1* and *img2*.  Each image has a voxel-to-world
transform associated with it.  (The "world" for these two transforms
could be similar or even identical in the case of an fmri series.)  I
would like to get from voxel coordinates in *img1* to voxel
coordinates in *img2*, for resampling::

  imgA = load_image(infile_A)
  vx2mmA = imgA.coordmap
  imgB = load_image(infile_B)
  vx2mmB = imgB.coordmap
  mm2vxB = vx2mmB.inverse
  # I want to first apply transform implied in 
  # cmA, then the inverse of transform implied in 
  # cmB.  If these are matrices then this would be
  # np.dot(mm2vxB, vx2mmA)
  voxA_to_voxB = mm2vxB.composewith(vx2mmA)

The (matrix) multiply version of this syntax would be::

  voxA_to_voxB = mm2vxB * vx2mmA
  
Composition should be of form ``Second.composewith(First)`` - as in
``voxA_to_voxB = mm2vxB.composewith(vx2mmA)`` above. The alternative
is ``First.composewith(Second)``, as in ``voxA_to_voxB =
vx2mmA.composewith(mm2vxB)``.  We choose ``Second.composewith(First)``
on the basis that people need to understand the mathematics of
function composition to some degree - see
wikipedia_function_composition_.

.. _wikipedia_function_composition: http://en.wikipedia.org/wiki/Function_composition

Real world to real world transform
''''''''''''''''''''''''''''''''''

We remind each other that a mapping is a function (callable) that takes
coordinates as input and returns coordinates as output.  So, if *M* is
a mapping then::

  [i',j',k'] = M(i, j, k)

where the *i, j, k* tuple is a coordinate, and the *i', j', k'* tuple is a
transformed coordinate.

Let us imagine we have somehow come by a mapping *T* that relates a
coordinate in a world space (mm) to other coordinates in a world
space.  A registration may return such a real-world to
real-world mapping.  Let us say that *V* is a useful mapping
matching the voxel coordinates in *img1* to voxel coordinates in
*img2*.  If *img1* has a voxel to mm mapping *M1* and *img2* has a mm
to voxel mapping of *inv_M2*, as in the previous example (repeated here)::

  imgA = load_image(infile_A)
  vx2mmA = imgA.coordmap
  imgB = load_image(infile_B)
  vx2mmB = imgB.coordmap
  mm2vxB = vx2mmB.inverse

then the registration may return the some coordinate map, *T* such that the
intended mapping *V* from voxels in *img1* to voxels in *img2* is::

  mm2vxB_map = mm2vxB.mapping
  vx2mmA_map = vx2mmA.mapping
  V = mm2vxB_map.composewith(T.composedwith(vx2mmA_map))

To support this, there should be a CoordinateMap constructor that
looks like this::

  T_coordmap = mm2mm(T)

where *T* is a mapping, so that::

  V_coordmap = mm2vxB.composewith(T_coordmap.composedwith(vx2mmA))



I have done a coregistration between two images, *img1* and *img2*.
This has given me a voxel-to-voxel transformation and I want to store
this transformation in such a way that I can use this transform to
resample *img1* to *img2*.  :ref:`resampling`

I have done a coregistration between two images, *img1* and *img2*. I
may want this to give me a worldA-to-worldB transformation, where
worldA is the world of voxel-to-world for *img1*, and worldB is the
world of voxel-to-world of *img2*.  

My *img1* has a voxel to world transformation.  This transformation
may (for example) have come from the scanner that acquired the image -
so telling me how the voxel positions in *img1* correspond to
physical coordinates in terms of the magnet isocenter and millimeters
in terms of the primary gradient orientations (x, y and z). I have the
same for *img2*.  For example, I might choose to display this image
resampled so each voxel is a 1mm cube.

Now I have these transformations:  ST(*img1*-V2W), and
ST(*img2*-V2W) (where ST is *scanner tranform* as above, and *V2W* is
voxel to world).

I have now done a coregistration between *img1* and *img2*
(somehow) - giving me, in addition to *img1* and *img2*, a
transformation that registers *img1* and *img2*. Let's call this
tranformation V2V(*img1*, *img2*), where V2V is voxel-to-voxel.

In actuality *img2* can be an array of images, such as series of fMRI
images and I want to align all the *img2* series to *img1* and then
take these voxel-to-voxel aligned images (the *img1* and *img2* array)
and remap them to the world space (voxel-to-world). Since remapping is
an interpolation operation I can generate errors in the resampled
pixel values. If I do more than one resampling, error will
accumulate. I want to do only a single resampling. To avoid the errors
associated with resampling I will build a *composite transformation*
that will chain the separate voxel-to-voxel and voxel-to-world
transformations into a single transformation function (such as an
affine matrix that is the result of multiplying the several affine
matrices together). With this single *composite transformatio* I now
resample *img1* and *img2* and put them into the world coordinate
system from which I can make measurements.

