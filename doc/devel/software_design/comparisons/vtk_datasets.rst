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

VTK datasets consist of two components:

* *organizing structure* - the topology and geometry
* *data attributes* - data that can be attached to the topology /
   geometry above.



Topology / geometry
-------------------

A dataset has one or more cells.


Geometry must first be specified with topology (the connectedness of the
data) and thence by the location of 

.. [VTK4]
   Schroeder, Will, Ken Martin, and Bill Lorensen. (2006) *The 
   Visualization Toolkit--An Object-Oriented Approach To 3D Graphics*. : 
   Kitware, Inc.


