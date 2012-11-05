.. basic_data_io:

===============
 Basic Data IO
===============

Accessing images using nipy:

While Nifti_ is the primary file format Analyze images (with associated .mat
file), and MINC files can also be read.

Load Image from File
====================

Get a filename for an example file.  ``anatfile`` gives a filename for a small
testing image in the nipy distribution:

>>> from nipy.testing import anatfile

Load the file from disk:

>>> from nipy import load_image
>>> myimg = load_image(anatfile)
>>> myimg.shape
(33, 41, 25)
>>> myimg.affine
array([[ -2.,   0.,   0.,  32.],
       [  0.,   2.,   0., -40.],
       [  0.,   0.,   2., -16.],
       [  0.,   0.,   0.,   1.]])

Access Data into an Array
=========================

This allows the user to access data as a numpy array.

>>> mydata = myimg.get_data()
>>> mydata.shape
(33, 41, 25)
>>> mydata.ndim
3

Save image to a File
====================

>>> from nipy import save_image
>>> newimg = save_image(myimg, 'newmyfile.nii')

Create Image from an Array
===========================

This will have a generic affine-type CoordinateMap with unit voxel sizes.

>>> import numpy as np
>>> from nipy.core.api import Image, vox2mni
>>> rawarray = np.zeros((43,128,128))
>>> arr_img = Image(rawarray, vox2mni(np.eye(4)))
>>> arr_img.shape
(43, 128, 128)

Coordinate map
==============

Images have a Coordinate Map.

The Coordinate Map contains information defining the input (domain) and output
(range) Coordinate Systems of the image, and the mapping between the two
Coordinate systems.  The *input* coordinate system is the *voxel* coordinate
system, and the *output* coordinate system is the *world* coordinate system.

>>> newimg.coordmap
AffineTransform(
   function_domain=CoordinateSystem(coord_names=('i', 'j', 'k'), name='voxels', coord_dtype=float64),
   function_range=CoordinateSystem(coord_names=('aligned-x=L->R', 'aligned-y=P->A', 'aligned-z=I->S'), name='aligned', coord_dtype=float64),
   affine=array([[ -2.,   0.,   0.,  32.],
                 [  0.,   2.,   0., -40.],
                 [  0.,   0.,   2., -16.],
                 [  0.,   0.,   0.,   1.]])

See :ref:`coordinate_map` for more detail.

.. include:: ../links_names.txt
