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

I have an array that represents voxels in an image and have a matrix
which represents the relation between the voxel coordinates and the
coordinates in scanner space.  I want to associate the array with the
matrix.

I have done a coregistration between two images, ImageA and ImageB.
This has given me a voxel-to-voxel transformation and I want to store
this transformation in such a way that I can use this transform to
resample ImageA to ImageB or vice versa.

I am going to do a coregistration between two images, ImageA and
ImageB.  Each image has a voxel-to-world transform.  The *world* for
these two transforms is similar, I want to use these two transforms to
put these images into *initial registration* for display or for
starting estimates in coregistration.

