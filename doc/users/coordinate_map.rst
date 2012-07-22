.. _coordinate_map:

#############################
 Basics of the Coordinate Map
#############################

When you load an image it will have an associated Coordinate Map

**Coordinate Map**

    The Coordinate Map contains information defining the input (domain) and
    output (range) Coordinate Systems of the image, and the mapping between the
    two Coordinate systems.

The *input* or *domain* in an image are voxel coordinates in the image array.
The *output* or *range* are the millimetre coordinates in some space, that
correspond to the input (voxel) coordinates.

>>> import nipy

Get a filename for an example file:

>>> from nipy.testing import anatfile

Get the coordinate map for the image:

>>> anat_img = nipy.load_image(anatfile)
>>> coordmap = anat_img.coordmap

For more on Coordinate Systems and thier properties
:mod:`nipy.core.reference.coordinate_system`

You can inspect a coordinate map::

>>> coordmap.function_domain.coord_names
>>> ('i', 'j', 'k')

>>> coordmap.function_range.coord_names
('aligned-x=L->R', 'aligned-y=P->A', 'aligned-z=I->S')

>>> coordmap.function_domain.name
'voxels'
>>> coordmap.function_range.name
'aligned'

A Coordinate Map has a mapping from the *input* Coordinate System to the
*output* Coordinate System

Here we can see we have a voxel to millimeter mapping from the voxel
space (i,j,k) to the millimeter space (x,y,z)

We can also get the name of the respective Coordinate Systems that our
Coordinate Map maps between.

A Coordinate Map is two Coordinate Systems with a mapping between
them.  Formally the mapping is a function that takes points from the
input Coordinate System and returns points from the output Coordinate
System.  This is the same as saying that the mapping takes points in the mapping
function *domain* and transforms them to points in the mapping function *range*.

Often this is simple as applying an Affine transform. In that case the
Coordinate System may well have an affine property which returns the
affine matrix corresponding to the transform.

>>> coordmap.affine
array([[ -2.,   0.,   0.,  32.],
       [  0.,   2.,   0., -40.],
       [  0.,   0.,   2., -16.],
       [  0.,   0.,   0.,   1.]])

If you call the Coordinate Map you will apply the mapping function
between the two Coordinate Systems. In this case from (i,j,k) to (x,y,z):

>>> coordmap([1,2,3])
array([ 30., -36., -10.])

It can also be used to  get the inverse mapping, or in this example from (x,y,z)
back to (i,j,k):

>>> coordmap.inverse()([30.,-36.,-10.])
array([ 1.,  2.,  3.])

We can see how this works if we just apply the affine
ourselves using dot product.

.. Note::

    Notice the affine is using homogeneous coordinates so we need to add a 1 to
    our input. (And note how  a direct call to the coordinate map does this work
    for you)

>>> coordmap.affine
array([[ -2.,   0.,   0.,  32.],
       [  0.,   2.,   0., -40.],
       [  0.,   0.,   2., -16.],
       [  0.,   0.,   0.,   1.]])

>>> import numpy as np
>>> np.dot(coordmap.affine, np.transpose([1,2,3,1]))
array([ 30., -36., -10.,   1.])

.. Note::

   The answer is the same as above (except for the added 1)

.. _normalize-coordmap:

***************************************************
Use of the Coordinate Map for spatial normalization
***************************************************

The Coordinate Map can be used to describe the transformations needed to perform
spatial normalization. Suppose we have an anatomical Image from one subject
*subject_img* and we want to create an Image in a standard space like Tailarach
space. An affine registration algorithm will produce a 4-by-4 matrix
representing the affine transformation, *T*, that takes a point in the subject's
coordinates *subject_world* to a point in Tailarach space *tailarach_world*. The
subject's Image has its own Coordinate Map, *subject_cmap* and there is a
Coordinate Map for Tailarach space which we will call *tailarach_cmap*.

Having found the transformation matrix *T*, the next step in spatial
normalization is usually to resample the array of *subject_img* so that it has
the same shape as some atlas *atlas_img*. Note that because it is an atlas
Image, *tailarach_camp=atlas_img.coordmap*.

A resampling algorithm uses an interpolator which needs to know
which voxel of *subject_img* corresponds to which voxel of *atlas_img*.
This is therefore a function from *atlas_voxel* to *subject_voxel*.

This function, paired with the information that it is a map from atlas-voxel to
subject-voxel is another example of a Coordinate Map. The code to do this might
look something like the following:

>>> from nipy.testing import anatfile, funcfile
>>> from nipy.algorithms.registration import HistogramRegistration
>>> from nipy.algorithms.kernel_smooth import LinearFilter

We'll make a smoothed version of the anatomical example image, and pretend it's
the template

>>> smoother = LinearFilter(anat_img.coordmap, anat_img.shape)
>>> atlas_im = smoother.smooth(anat_img)
>>> subject_im = anat_img

We do an affine registration between the two.

>>> reggie = HistogramRegistration(subject_im, atlas_im)
>>> aff = reggie.optimize('affine').as_affine() #doctest: +ELLIPSIS
Initial guess...
...

Now we make a coordmap with this transformation

>>> from nipy.core.api import AffineTransform
>>> subject_cmap = subject_im.coordmap
>>> talairach_cmap = atlas_im.coordmap
>>> subject_world_to_talairach_world = AffineTransform(
...                                       subject_cmap.function_range,
...                                       talairach_cmap.function_range,
...                                       aff)
...

We resample the 'subject' image to the 'atlas image

>>> from nipy.algorithms.resample import resample
>>> normalized_subject_im = resample(subject_im, talairach_cmap,
...                                  subject_world_to_talairach_world,
...                                  atlas_im.shape)
>>> normalized_subject_im.shape == atlas_im.shape
True
>>> normalized_subject_im.coordmap == atlas_im.coordmap
True
>>> np.all(normalized_subject_im.affine == atlas_im.affine)
True

***********************
Mathematical definition
***********************

For a more formal mathematical description of the coordinate map, see
:ref:`math-coordmap`.
