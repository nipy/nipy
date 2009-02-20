.. basic_data_io:

===============
 Basic Data IO
===============

Accessing images using nipy:

Nifti_ is the primary file format

Load Image from File
====================

.. sourcecode::  ipython

  from neuroimaging.core.api import load_image
  infile = 'myimage.nii'
  myimg = load_image(infile)


Access Data into an Array
=========================

This allows user to access data in a numpy array. 

.. Note::

   This is the correct way to access the data as it applies the proper
   intensity scaling to the image as defined in the header

.. sourcecode::  ipython

   from neuroimaging.core.api import load_image
   import numpy as np
   myimg = load_file('myfile')
   mydata = np.asarray(myimg)
   mydata.shape

Save image to a File
====================

.. sourcecode::  ipython

   from neuroimaging.core.api import load_image,save_image
   import numpy as np
   myimg = load_file('myfile.nii')	
   newimg = save_file(myimg,'newmyfile.nii')
   

Create Image from an Array
===========================

This will have a generic CoordinateMap with Unit step sizes

.. sourcecode::  ipython

   from neuroimaging.core.api import fromarray, save_image
   import numpy as np
   rawarray = np.zeros(43,128,128)
   innames='kij'
   outnames='zyx'
   newimg = fromarray(rawarray, innames, outnames)


Images have a Coordinate Map.

The Coordinate Map contains information defining the input and output
Coordinate Systems of the image, and the mapping between the two
Coordinate systems.

:ref:`coordinate_map`

Here is an examples file image_fromarray.py that shows the use of io
and Coordinate Maps.

.. literalinclude:: ../../examples/image_fromarray.py

.. include:: ../links_names.txt
