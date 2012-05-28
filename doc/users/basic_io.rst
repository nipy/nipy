.. basic_data_io:

===============
 Basic Data IO
===============

Accessing images using nipy:

While Nifti_ is the primary file format Analyze images (with
associated .mat file), and MINC files can also 
be read.

Load Image from File
====================

.. sourcecode::  ipython

  from nipy.io.api import load_image
  infile = 'myimage.nii'
  myimg = load_image(infile)
  myimg.shape
  myimg.affine

Access Data into an Array
=========================

This allows user to access data in a numpy array. 

.. Note::

   This is the correct way to access the data as it applies the proper
   intensity scaling to the image as defined in the header

.. sourcecode::  ipython

   from nipy.io.api import load_image
   import numpy as np
   myimg = load_file('myfile')
   mydata = np.asarray(myimg)
   mydata.shape

Save image to a File
====================

.. sourcecode::  ipython

   from nipy.io.api import load_image,save_image
   import numpy as np
   myimg = load_file('myfile.nii')
   newimg = save_image(myimg,'newmyfile.nii')


Create Image from an Array
===========================

This will have a generic affine-type CoordinateMap with Unit step sizes

.. sourcecode::  ipython

   from nipy.core.api import Image, AffineTransform
   from nipy.io.api import save_image
   import numpy as np
   rawarray = np.zeros(43,128,128)
   innames='ijk'
   outnames='xyz'
   mapping = np.eye(4)
   newimg = Image(rawarray, AffineTransform(innames, outnames, mapping))

Images have a Coordinate Map.

The Coordinate Map contains information defining the input and output
Coordinate Systems of the image, and the mapping between the two
Coordinate systems.

:ref:`coordinate_map`


.. include:: ../links_names.txt
