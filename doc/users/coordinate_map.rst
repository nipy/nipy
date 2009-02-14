.. _coordinate_map:

==============================
 Basics of the Coordinate Map
==============================

When you load an image it will have an associated Coordinate Map

.. sourcecode::  ipython

  from neuroimaging.core.api import load_image
  infile = 'Talairach-labels-1mm.nii'
  myimg = load_image(infile)
  coordmap = myimg.coordmap

The Coordinate Map contains information defining the input and output
Coordinate Systems of the image, and the mapping between the two
Coordinate systems.

For more on Coordinate Systems, Coordinates and thier properties
:mod:`neuroimaging.core.reference.coordinate_system`

You can introspect a coordinate map

.. sourcecode::  ipython

   In [5]: coordmap.input_coords
   Out[5]: {'axes': [('k', <Coordinate:"k", dtype=[('k', '<f8')]>),
   ('j', <Coordinate:"j", dtype=[('j', '<f8')]>), 
   ('i', <Coordinate:"i", dtype=[('i', '<f8')]>)],
    'name': 'input-reordered'}
   In [7]: coordmap.input_coords.axisnames
   Out[7]: ['k', 'j', 'i']

   In [8]: coordmap.output_coords
   Out[8]: {'axes': [('z', <Coordinate:"z", dtype=[('z', '<f8')]>), 
   ('y', <Coordinate:"y", dtype=[('y', '<f8')]>), 
   ('x', <Coordinate:"x", dtype=[('x', '<f8')]>)], 
   'name': 'output-reordered'}
   In [9]: coordmap.output_coords.axisnames
   Out[9]: ['z', 'y', 'x']

.. Note::

   People using matlab are used to seeing the Coordinate System
   (i,j,k) mapping to the Coordinate System (x,y,z). This is due to
   fortran ordered reading of the data.
   
   Numpy's default ordering is C ordered which is why in this case (k,j,i)
   maps to (z,y,x)

A Coordinate Map has a mapping from the *input* Coordinate System to the
*output* Coordinate System

Here we can see we have a voxel to millimeter mapping from the voxel
space (k,j,i) to the millimeter space (z,y,x)

We also know from the dtype attribute that the Axes in this
Coordinate System are of type '<f8' *little-endian float64*

   
We can also get the name of the respective Coordinate Systems that our
Coordinate Map maps between

.. sourcecode::  ipython

   In [20]: coordmap.input_coords.name
   Out[20]: 'input-reordered'

   In [19]: coordmap.output_coords.name
   Out[19]: 'output-reordered'


A Coordinate Map is two Coordinate Systems with a mapping between
them.  Formally the mapping is a function that takes points from the
input Coordinate System and returns points from the output Coordinate
System.

Often this is simple as applying an Affine transform. In that case the
Coordinate System may well have an affine property which returns the
affine matrix corresponding to the transform. 

.. sourcecode::  ipython

   In [11]: coordmap.affine
   Out[11]: 
   array([[   1.,    0.,    0.,  -72.],
          [   0.,    1.,    0., -126.],
          [   0.,    0.,   -1.,   90.],
          [   0.,    0.,    0.,    1.]])


If you call the Coordinate Map you will apply the mapping function
between the two Coordinate Systems. In this case from (k,j,i) to (z,y,x)

.. sourcecode::  ipython

   In [13]: coordmap([1,2,3])
   Out[13]: array([ -71., -124.,   87.])


It can also be used to  get the inverse mapping, or in this 
example from (z,y,x) back to (k,j,i)

.. sourcecode::  ipython

   In [14]: coordmap.inverse_mapping([-71.,-124.,87.])
   Out[14]: array([ 1.,  2.,  3.])


We can see how this works if we just apply the affine
ourselves. Notice the affine is using homogeneous coordinates so we
need to add a 1 to our input. (And note how  a direct call to the coordinate map does
this work for you)

.. sourcecode::  ipython

   In [15]: coordmap.affine
   Out[15]: 
   array([[   1.,    0.,    0.,  -72.],
          [   0.,    1.,    0., -126.],
          [   0.,    0.,   -1.,   90.],
          [   0.,    0.,    0.,    1.]])

    In [17]: import numpy as np

    In [18]: np.dot(coordmap.affine, np.transpose([1,2,3,1]))
    Out[18]: array([ -71., -124.,   87.,    1.])

.. Note::

   The answer is the same as above (except for the added 1)
