.. _image_ordering:

Image index ordering
====================

Background
----------

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
memory map it:

::

   img_arr = memmap('my.img', dtype=float32)

When we index this new array, the first index indexes the Z dimension,
and the third indexes X.  For example, if I want a voxel X=3, Y=10,
Z=20 (zero-based), I have to get this from the array with:

::

   img_arr[20, 10, 3]


The problem
-----------

Most potential users of NiPy are likely to have experience of using
image arrays in Matlab and SPM.  Matlab uses Fortran index ordering.
For fortran, the first index is the fastest changing, and the last is
the slowest-changing. For example, here is how to get voxel X=3, Y=10,
Z=20 (zero-based) using SPM in Matlab:

::

   img_arr = spm_read_vols(spm_vol('my.img'));
   img_arr(4, 11, 21)  % matlab indexing is one-based


This ordering fits better with the way that we talk about coordinates
in functional imaging, as we invariably use XYZ ordered coordinates in
papers.  It is possible to do the same in numpy, by specifying that
the image should have fortran index ordering:

::

   img_arr = memmap('my.img', dtype=float32, order='F')
   img_arr[3, 10, 20]


Native fortran or C indexing for images
---------------------------------------

We could change the default ordering of image arrays to fortran, in
order to allow XYZ index ordering.  So, change the access to the image
array in the image class so that, to get the voxel at X=3, Y=10, Z=20
(zero-based):

::

   img = load_image('my.img')
   img[3, 10, 20]


instead of the current situation, which requires:

::

   img = load_image('my.img')
   img[20, 10, 3]


For and against fortran ordering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For:

* Fortran index ordering is more intuitive for functional imaging
  because of conventional XYZ ordering of spatial coordinates, and
  Fortran index ordering in packages such as Matlab
* Indexing into a raw array is fast, and common in lower-level
  applications, so it would be useful to implement the more intuitive
  XYZ ordering at this level rather than via interpolators (see below)
* Standardizing to one index ordering (XYZ) would mean users would not
  have to think about the arrangement of the image in memory

Against:

* C index ordering is more familiar to C users
* C index ordering is the default in numpy
* XYZ ordering can be implemented by wrapping by an interpolator

Note that there is no performance penalty for either array ordering,
as this is dealt with internally by NumPy.  For example, imagine the
following::

   arr = np.empty((100,50)) # Indexing is C by default
   arr2 = arr.transpose() # Now it is fortran
   # There should be no effective difference in speed for the next two lines
   b = arr[0] # get first row of data - most discontiguous memory
   c = arr2[:,0] # gets same data, again most discontiguous memory

Potential problems for fortran ordering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clash between default ordering of numpy arrays and nipy images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

C index ordering is the default in numpy, and using fortran ordering
for images might be confusing in some circumstances.  Consider for
example:

::

   img_obj = load_image('my.img') # Where the Image class has been changed to implement Fortran ordering
   first_z_slice = img_obj[...,0] # returns a Z slice
   
   img_arr = memmap('my.img', dtype=float32) # C ordering, the numpy default
   img_obj = Image.from_array(img_arr) # this call may not be correct
   first_z_slice = img_obj[...,0]  # in fact returns an X slice


I suppose that we could check that arrays are fortran index ordered in
the Image __init__ routine.

An alternative proposal - XYZ ordering of output coordinates
------------------------------------------------------------

JT: Another thought, that is a compromise between the XYZ coordinates
and Fortran ordering.

To me, having worked mostly with C-type arrays, when I index an array
I think in C terms. But, the Image objects have the "warp" attached to
them, which describes the output coordinates. We could insist that the
output coordinates are XYZT (or make this an option). So, for
instance, if the 4x4 transform was the identity, the following two
calls would give something like:

::

    >>> interp = interpolator(img)
    >>> img[3,4,5] == interp(5,4,3)
   True


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
