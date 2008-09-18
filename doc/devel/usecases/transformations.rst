.. _transformations:

==========================
 Transformation use cases
==========================

Use cases for defining and using transforms on images.


I have an image **Img**.  I would like to know what the voxel sizes
are.  I would like to determine whether it was acquired axially,
coronally or sagittally.  What is the brain orientation in relation to
the voxels?  Has it been acquired at an oblique angle?  What are the
voxel dimensions?

I have an array that represents voxels in an image and have a
matrix/transform which represents the relation between the voxel
coordinates and the coordinates in scanner space, *world coordinates*.
I want to associate the array with the matrix.

I have two images, ImageA and ImageB.  Each image has a voxel-to-world
transform associated with it.  (The *world* for these two transforms
could be similar or even identical in the case of an fmri series.)  I would
like to get from voxel coordinates in ImageA to voxel coordinates in
ImageB.  This would result in a voxel-to-voxel transform.  (This is a
rigid-body transformation.)

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

