.. _understanding_affines:

=============================================
 Understanding voxel and real world mappings
=============================================

Voxel coordinates and real-world coordinates
----------------------------------------------

A point can be represented by coordinates relative to specified axes.
coordinates are (almost always) numbers - see `coordinate systems
<http://en.wikipedia.org/wiki/Coordinate_system>`_

For example, a map grid reference gives a coordinate (a pair of
numbers) to a point on the map.  The numbers give the respective
positions on the horizontal (``x``) and vertical (``y``) axes of the
map.

A coordinate system is defined by a set of axes.  In the example
above, the axes are the ``x`` and ``y`` axes.  Axes for coordinates
are usually orthogonal - for example, moving one unit up on the ``x``
axis on the map causes no change in the ``y`` coordinate - because
the axes are at 90 degrees.  

In this discussion we'll concentrate on the three dimensional case.
Having three dimensions means that we have a three axis coordinate
system, and coordinates have three values.  The meaning of the values
depend on what the axes are.

Voxel coordinates
`````````````````

Array indexing is one example of using a coordinate system.  Let's say
we have a three dimensional array::

  A = np.arange(24).reshape((2,3,4))

The value ``0`` is at array coordinate ``0,0,0``::

  assert A[0,0,0] == 0

and the value ``23`` is at array coordinate ``1,2,3``::

  assert A[1,2,3] == 23

(remembering python's zero-based indexing). If we now say that our
array is a 3D volume element array - an array of voxels, then
the array coordinate is also a voxel coordinate.

If we want to use ``numpy`` to index our array, then we need integer
voxel coordinates, but if we use a resampling scheme, we can also
imagine non-integer voxel coordinates for ``A``, such as
``(0.6,1.2,1.9)``, and we could use resampling to estimate the value
at such a coordinate, given the actual data in the surrounding
(integer) points.

Array / voxel coordinates refer to the array axes.  Without any
further information, they do not tell us about where the point is in
the real world - the world we can measure with a ruler.  We refer to
array / voxel coordinates with indices ``i, j, k``, where ``i`` is the
first value in the 3 value coordinate tuple. For example, if array /
voxel point ``(1,2,3)`` has ``i=1, j=2, k=3``.  We'll be careful only
to use ``i, j, k`` rather than ``x, y, z``, because we are going to
use ``x, y, z`` to refer to real-world coordinates. 

Real-world coordinates
``````````````````````
Real-world coordinates are coordinates where the values refer to
real-world axes.  A real-world axis is an axis that refers to some
real physical space, like low to high position in an MRI scanner, or
the position in terms of the subject's head.

Here we'll use the usual neuroimaging convention, and that is to label
our axes relative to the subject's head:

 * ``x`` has negative values for left and positive values for right
 * ``y`` has negative values for posterior (back of head) and positive
   values for anterior (front of head)
 * ``z`` has negative values for the inferior (towards the neck) and
   postive values for superior (towards the highest point of the head,
   when standing)

Image index ordering
--------------------

Background
``````````

In general, images - and in particular NIfTI format images, are
ordered in memory with the X dimension changing fastest, and the Z
dimension changing slowest.

Numpy has two different ways of indexing arrays in memory, C and
fortran.  With C index ordering, the first index into an array indexes
the slowest changing dimension, and the last indexes the fastest
changing dimension.  With fortran ordering, the first index refers to
the fastest changing dimension - X in the case of the image mentioned
above.

C is the default index ordering for arrays in Numpy. 

For example, let's imagine that we have a binary block of 3D image
data, in standard NIfTI / Analyze format, with the X dimension
changing fastest, called `my.img`, containing Float32 data.  Then we
memory map it::

  img_arr = memmap('my.img', dtype=float32)

When we index this new array, the first index indexes the Z dimension, and the third indexes X.  For example, if I want a voxel X=3, Y=10, Z=20 (zero-based), I have to get this from the array with::

  img_arr[20, 10, 3]

The problem
```````````

Most potential users of NiPy are likely to have experience of using
image arrays in Matlab and SPM.  Matlab uses Fortran index ordering.
For fortran, the first index is the fastest changing, and the last is
the slowest-changing. For example, here is how to get voxel X=3, Y=10,
Z=20 (zero-based) using SPM in Matlab::

  img_arr = spm_read_vols(spm_vol('my.img'));
  img_arr(4, 11, 21)  % matlab indexing is one-based

This ordering fits better with the way that we talk about coordinates
in functional imaging, as we invariably use XYZ ordered coordinates in
papers.  It is possible to do the same in numpy, by specifying that
the image should have fortran index ordering::

  img_arr = memmap('my.img', dtype=float32, order='F')
  img_arr[3, 10, 20]

The proposal
````````````

Change the default ordering of image arrays to fortran, in order to
allow XYZ index ordering.  So, change the access to the image array in
the image class so that, to get the voxel at X=3, Y=10, Z=20
(zero-based)::

  img = Image('my.img')
  img[3, 10, 20]

instead of the current situation, which requires::

  img = Image('my.img')
  img[20, 10, 3]

Summary of discussion
`````````````````````

For:

 * Fortran index ordering is more intuitive for functional imaging because of conventional XYZ ordering of spatial coordinates, and Fortran index ordering in packages such as Matlab
 * Indexing into a raw array is fast, and common in lower-level applications, so it would be useful to implement the more intuitive XYZ ordering at this level rather than via interpolators (see below)
 * Standardizing to one index ordering (XYZ) would mean users would not have to think about the arrangement of the image in memory

Against:

 * C index ordering is more familiar to C users
 * C index ordering is the default in numpy
 * XYZ ordering can be implemented by wrapping by an interpolator 

Potential problems
``````````````````

Performance penalties
^^^^^^^^^^^^^^^^^^^^^

KY commented:: 

  This seems like a good idea to me but I have no knowledge of numpy
  internals (and even less than none after the numeric/numarray
  integration). Does anyone know if this will (or definitely will not)
  incur any kind of obvious performance penalties re. array operations
  (sans arcane problems like stride issues in huge arrays)?

MB replied:

  Note that, we are not proposing to change the memory layout of the
  image, which is fixed by the image format in e.g NIfTI, but only to
  index it XYZ instead of ZYX.  As far as I am aware, there are no
  significant performance differences between::

    img_arr = memmap('my.img', dtype=float32, order='C')
    img_arr[5,4,3]

  and::

    img_arr = memmap('my.img', dtype=float32, order='F')
    img_arr[3,4,5]

  Happy to be corrected though.  

Clash between default ordering of numpy arrays and nipy images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

C index ordering is the default in numpy, and using fortran ordering
for images might be confusing in some circumstances.  Consider for
example:

  img_obj = Image('my.img') # Where the Image class has been changed to implement Fortran ordering
  first_z_slice = img_obj[...,0] # returns a Z slice

  img_arr = memmap('my.img', dtype=float32) # C ordering, the numpy default
  img_obj = Image(img_arr)
  first_z_slice = img_obj[...,0]  # in fact returns an X slice

I suppose that we could check that arrays are fortran index ordered in the Image __init__ routine. 

An alternative proposal - XYZ ordering of output coordinates
````````````````````````````````````````````````````````````
JT: Another thought, that is a compromise between the XYZ coordinates and Fortran ordering.

To me, having worked mostly with C-type arrays, when I index an array
I think in C terms. But, the Image objects have the "warp" attached to
them, which describes the output coordinates. We could insist that the
output coordinates are XYZT (or make this an option). So, for
instance, if the 4x4 transform was the identity, the following two
calls would give something like::

  interp = interpolator(img)
  img[3,4,5] == interp(5,4,3)

This way, users would be sure in the interpolator of the order of the
coordinates, but users who want access to the array would know that
they would be using the array order on disk...

I see that a lot of users will want to think of the first coordinate
as "x", but depending on the sampling the [0] slice of img may be the
leftmost or the rightmost. To find out which is which, users will have
to look at the 4x4 transform (or equivalently the start and the
step). So just knowing the first array coordinate is the "x"
coordinate still misses some information, all of which is contained in
the transform.

MB replied:

I agree that the output coordinates are very important - and I think
we all agree that this should be XYZ(T)?

For the raw array indices - it is very common for people to want to do
things to the raw image array - the quickstart examples containing a
few - and you usually don't care about which end of X is left in that
situation, only which spatial etc dimension the index refers to.
