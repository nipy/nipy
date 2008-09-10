==============
 Nipy roadmap
==============

Registration
------------

Matthew wants:

- A collection of standard images for testing on, from different
  scanners, with different levels of noise, or movement.  Possibly
  lesions.  Provenance and licensing from get-go.  Maybe some fruit.

Tom's plans / work:

Current state is:

- Actual code for rigid body registration with a variety of cost
  functions is in `neuroimaging.fixes.scipy.ndimage`.  Methods take
  two ndarrays as input for registration, with associated affine
  mappings into real space.
- Various higher-level wrapper functions for automating movement
  correction, anatomical to functional registration.  This is
  currently in `lp:~twaite/+junk/registration`.
- Segmentation similarly in `fixes.scipy.ndimage`, basic routines,
  whereas some higher level and more specialized code in
  `lp:~twaite/+junk/segmentation`.
- There is some normalization code in the
  `lp:~twaite/+junk/registration` in `normz`.  This is a port of the
  SPM linear and non-linear normalization.

Plan is:

#. Complete testing of moving interpolation to Unser spline algorithm
   (implemented already in `ndimage` interpolation) - from Tom's custom
   interpolation.
#. Put Tom's custom cubic interpolation algorithm somewhere for
   reference, and because it is quick.
#. Complete / integrate rigid body registration into nipy_, by using
   nipy_ image IO, image registration.  This is work in
   `lp:~twaite/+junk/registration`.  It will probably end up somewhere
   like `neuroimaging.algorithms.spatialprocessing`. 
#. Review general framework for spatial normalization code (for
   documentation and better code understanding)
#. Refactor SPM elastic deformation code for greater clarity and maybe
   greater potential for parallel implementation. 
#. Design test case images / arrays for components of spatial
   normalization - phantoms for example.
#. In due course we need to think more carefully about validation
   algorithms. 

In long term:

#. HAMMER with Karl Young


.. _nipy: https://launchpad.net/nipy

   
