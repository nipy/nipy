==============
 VTK datasets
==============

Here we describe the VTK dataset model, because of some parallels with
our own idea of an image object.  The document is from the VTK book - [VTK4]_

See also: 

* http://code.enthought.com/projects/mayavi/docs/development/html/mayavi/data.html#vtk-data-structures 
* http://code.enthought.com/projects/mayavi/docs/development/html/mayavi/auto/example_datasets.html
* http://www.vtk.org/Wiki/VTK/Writing_VTK_files_using_python
* http://www.vtk.org/VTK/img/file-formats.pdf
* https://svn.enthought.com/enthought/attachment/wiki/MayaVi/tvtk_datasets.pdf?format=raw

What is a VTK dataset?
======================

VTK datasets represent discrete spatial data.

Datasets consist of two components:

* *organizing structure* - the topology and geometry
* *data attributes* - data that can be attached to the topology /
   geometry above.

Structure: topology / geometry
------------------------------

A dataset has one more points.  Points have coordinates in 3 dimensions,
in the order ``x``, ``y``, ``z`` - see http://www.vtk.org/doc/release/5.4/html/a00374.html - ``GetPoint()``

>>> import vtk
>>> pts = vtk.vtkPoints()
>>> pts.InsertNextPoint(0, 0, 0)
>>> pts.GetPoint(0)
(0.0, 0.0, 0.0)

A cell is a local specification of the connection between points - an
atom of topology in VTK.  A cell has a type, and a list of points.  

The type gives the type of local topology, and therefore how the list of
points are connected.  For example:


.. [VTK4]
   Schroeder, Will, Ken Martin, and Bill Lorensen. (2006) *The 
   Visualization Toolkit--An Object-Oriented Approach To 3D Graphics*. : 
   Kitware, Inc.


