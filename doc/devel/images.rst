===================
 Describing images
===================

Here we set out what we think an image is and how it should work in our
code.  We are largely following the nifti_ standard.

What is an image?
=================

An image is the association of a block (array) of spatial data, with the
relationship of the position of that data to some continuous space.

Therefore an image contains:

* an array
* a spatial transformation describing the position of the data in the
  array relative to some space.

An image always has 3 spatial dimensions.  It can have other dimensions,
such as time.

A slice from a 3D image is also a 3D image, but with one dimension of
the image having length 1.

The transformation is spatial and refers to exactly three dimensions.

::

    import numpy as np
    import neuroimaging as ni
    img = ni.load_image('example3d.img')
    arr = img.get_data()
    assert isinstance(arr, np.ndarray)
    xform = img.get_transform()
    voxel_position = [0, 0, 0]
    world_position = xform.apply(voxel_position)
    assert world_position.shape = (3,)

An image has an array.  The first 3 axes (dimensions) of that array are
spatial.  Further dimensions can have various meanings.  The most common
meaning of the 4th axis is time.

The relationship of the first three dimensions to any particular
orientation in space are only known from the image transform.

.. include:: ../links_names.txt
