=========================
 Registration API Design
=========================

This contains design ideas for the end-user api when registering images in nipy.

We want to provide a simple api, but with enough flexibility to allow
users to changes various components of the pipeline.  We will also
provide various **Standard** scripts that perform typical pipelines.

The pluggable script::

  func_img = load_image(filename)
  anat_img = load_image(filename)
  interpolator = SplineInterpolator(order=3)
  metric = NormalizedMutualInformation()
  optimizer = Powell()
  strategy = RegistrationStrategy(interpolator, metric, optimizer)
  w2w = strategy.apply(img_fixed, img_moving)

To apply the transform and resample the image::

  new_img = resample(img_moving, w2w, interp=interpolator)

Or::

  new_img = Image(img_moving, w2w*img_moving.coordmap)

Transform Multiplication
------------------------

The multiplication order is important and coordinate systems must
*make sense*.  The *output coordinates* of the mapping on the
right-hand of the operator, must match the *input coordinates* of the
mapping on the left-hand side of the operator.

For example, imageA has a mapping from voxels-to-world (v2w), imageB
has a mapping from world-to-world (w2w).  So the output of imageA,
*world*, maps to the input of imageB, *world*.  We would compose a new
mapping (transform) from these mappings like this::

  new_coordmap = imageB.coordmap * imageA.coordmap

If one tried to compose a mapping in the other order, an error should
be raised as the code would detect a mismatch of trying to map output
coordinates from imageB, *world* to the input coordinates of imageA,
*voxels*::

  new_coordmap = imageA.coordmap * imageB.coordmap
  raise ValueError!!!

Note: We should consider a meaningful error message to help people
quickly correct this mistake.

One way to remember this ordering is to think of composing functions.
If these were functions, the output of the first function to evaluate
(imageA.coordmap) is passed as input to the second function
(imageB.coordmap).  And therefore they must match::

  new_coordmap = imageB.coordmap(imageA.coordmap())

Matching Coordinate Systems
---------------------------

We need to make sure we can detect mismatched coordinate mappings.
The CoordinateSystem class has a check for equality (__eq__ method)
based on the axis and name attributes.  Long-term this may not be
robust enough, but it's a starting place.  We should write tests for
failing cases of this, if they don't already exists.

CoordinateMap
-------------

Recall the CoordinateMap defines a mapping between two coordinate
systems, an input coordinate system and an output coordinate system.
One example of this would be a mapping from voxel space to scanner
space.  In a Nifti1 header we would have an affine transform to apply
this mapping.  The *input coordinates* would be voxel space, the
*output coordinates* would be world space, and the affine transform
provides the mapping between them.

