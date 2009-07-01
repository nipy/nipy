.. _coordinate_map:

==============================
 Basics of the Coordinate Map
==============================

When you load an image it will have an associated Coordinate Map

**Coordinate Map** 

	     The Coordinate Map contains information defining the input and output
	     Coordinate Systems of the image, and the mapping between the two
	     Coordinate systems.


.. sourcecode::  ipython

  from nipy import load_image
  infile = 'Talairach-labels-1mm.nii'
  myimg = load_image(infile)
  coordmap = myimg.coordmap


For more on Coordinate Systems, Coordinates and thier properties
:mod:`neuroimaging.core.reference.coordinate_system`

You can introspect a coordinate map

.. sourcecode::  ipython

   In [15]: coordmap.input_coords.coord_names
   Out[15]: ('i', 'j', 'k')
  
  In [17]: coordmap.output_coords.coord_names
  Out[17]: ('x', 'y', 'z')

  In [18]: coordmap.affine
  Out[18]: 
  array([[  -1.,    0.,    0.,   90.],
  	    [   0.,    1.,    0., -126.],
            [   0.,    0.,    1.,  -72.],
            [   0.,    0.,    0.,    1.]])

  In [26]: coordmap.input_coords.name
  Out[26]: 'input'

  In [27]: coordmap.output_coords.name
  Out[27]: 'output'

  
A Coordinate Map has a mapping from the *input* Coordinate System to the
*output* Coordinate System

Here we can see we have a voxel to millimeter mapping from the voxel
space (i,j,k) to the millimeter space (x,y,z)

  
We can also get the name of the respective Coordinate Systems that our
Coordinate Map maps between


A Coordinate Map is two Coordinate Systems with a mapping between
them.  Formally the mapping is a function that takes points from the
input Coordinate System and returns points from the output Coordinate
System.

Often this is simple as applying an Affine transform. In that case the
Coordinate System may well have an affine property which returns the
affine matrix corresponding to the transform. 

.. sourcecode::  ipython

   In [31]: coordmap.affine
   Out[31]: 
   array([[  -1.,    0.,    0.,   90.],
       [   0.,    1.,    0., -126.],
       [   0.,    0.,    1.,  -72.],
       [   0.,    0.,    0.,    1.]])



If you call the Coordinate Map you will apply the mapping function
between the two Coordinate Systems. In this case from (i,j,k) to (x,y,z)

.. sourcecode::  ipython

   In [32]: coordmap([1,2,3])
   Out[32]: array([[  89., -124.,  -69.]])

   In [33]: coordmap.mapping([1,2,3])
   Out[33]: array([  89., -124.,  -69.])


It can also be used to  get the inverse mapping, or in this 
example from (x,y,z) back to (i,j,k)

.. sourcecode::  ipython

   In [35]: coordmap.inverse_mapping([89,-124,-69])
   Out[35]: array([ 1.,  2.,  3.])



We can see how this works if we just apply the affine
ourselves using dot product. 

.. Note::
    
	Notice the affine is using homogeneous coordinates so we
	need to add a 1 to our input. (And note how  a direct call to the 
	coordinate map does this work for you)

.. sourcecode::  ipython

   In [36]: coordmap.affine
   Out[36]: 
   array([[  -1.,    0.,    0.,   90.],
       [   0.,    1.,    0., -126.],
       [   0.,    0.,    1.,  -72.],
       [   0.,    0.,    0.,    1.]])

    In [37]: import numpy as np

    In [39]: np.dot(coordmap.affine,np.transpose([1,2,3,1]))
    Out[39]: array([  89., -124.,  -69.,    1.])

    
.. Note::

   The answer is the same as above (except for the added 1)
