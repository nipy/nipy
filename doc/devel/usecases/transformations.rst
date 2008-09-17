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

