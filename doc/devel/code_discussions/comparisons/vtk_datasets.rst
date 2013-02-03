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
* http://public.kitware.com/cgi-bin/viewcvs.cgi/*checkout*/Examples/DataManipulation/Python/BuildUGrid.py?root=VTK&content-type=text/plain

What is a VTK dataset?
======================

VTK datasets represent discrete spatial data.

Datasets consist of two components:

* *organizing structure* - the topology and geometry
* *data attributes* - data that can be attached to the topology /
   geometry above.

Structure: topology / geometry
------------------------------

The structure part of a dataset is the part that gives the position and
connection of points in 3D space.

Let us first import *vtk* for our code examples.

>>> import vtk

An *id* is an index into a given vector
---------------------------------------

We introduce *id* to explain the code below.  An id is simply an index
into a vector, and is therefore an integer.  Of course the id identifies
the element in the vector; as long as you know which vector the id
refers to, you can identify the element. 

>>> pts = vtk.vtkPoints()
>>> id = pts.InsertNextPoint(0, 0, 0)
>>> id == 0
True
>>> id = pts.InsertNextPoint(0, 1, 0)
>>> id == 1
True
>>> pts.GetPoint(1) == (0.0, 1.0, 0.0)
True

A dataset has one or more points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Points have coordinates in 3 dimensions, in the order ``x``, ``y``,
``z`` - see http://www.vtk.org/doc/release/5.4/html/a00374.html -
``GetPoint()``

>>> pts = vtk.vtkPoints()
>>> pts.InsertNextPoint(0, 0) # needs 3 coordinates
Traceback (most recent call last):
   ...
TypeError: function takes exactly 3 arguments (2 given)
>>> _ = pts.InsertNextPoint(0, 0, 0) # returns point index in point array
>>> pts.GetPoint(0)
(0.0, 0.0, 0.0)
>>> _ = pts.InsertNextPoint(0, 1, 0)
>>> _ = pts.InsertNextPoint(0, 0, 1)

A dataset has one or more cells
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A cell is a local specification of the connection between points - an
atom of topology in VTK.  A cell has a type, and a list of point ids.
The point type determines (by convention) what the connectivity of the
list of points should be.  For example we can make a cell of type
``vtkTriangle``.  The first point starts the triangle, the next point is
the next point in the triangle counterclockwise, connected to the first
and third, and the third is the remaining point, connected to the first
and second.

>>> VTK_TRIANGLE = 5 # A VTK constant identifying the triangle type
>>> triangle = vtk.vtkTriangle()
>>> isinstance(triangle, vtk.vtkCell)
True
>>> triangle.GetCellType() == VTK_TRIANGLE
True
>>> pt_ids = triangle.GetPointIds() # these are default (zeros) at the moment
>>> [pt_ids.GetId(i) for i in range(pt_ids.GetNumberOfIds())] == [0, 0, 0]
True

Here we set the ids.  The ids refer to the points above.  The system
does not know this yet, but it will because, later, we are going to
associate this cell with the points, in a dataset object.

>>> for i in range(pt_ids.GetNumberOfIds()): pt_ids.SetId(i, i)

Associating points and cells
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We make the most general possible of VTK datasets - the unstructured
grid.

>>> ugrid = vtk.vtkUnstructuredGrid()
>>> ugrid.Allocate(1, 1)
>>> ugrid.SetPoints(pts)
>>> id = ugrid.InsertNextCell(VTK_TRIANGLE, pt_ids)

Data attributes
---------------

So far we have specified a triangle, with 3 points, but no associated data.

You can associate data with cells, or with points, or both.  Point data
associates values (e.g. scalars) with the points in the dataset.  Cell
data associates values (e.g. scalars) with the cells - in this case one
(e.g) scalar value with the whole triangle.

>>> pt_data = ugrid.GetPointData()
>>> cell_data = ugrid.GetCellData()

There are many data attributes that can be set, include scalars,
vectors, normals (normalized vectors), texture coordinates and tensors,
using (respectively)
``{pt|cell|_data.{Get|Set}{Scalars|Vectors|Normals|TCoords|Tensors}``.
For example:

>>> pt_data.GetScalars() is None
True

But we can set the scalar (or other) data:

>>> tri_pt_data = vtk.vtkFloatArray()
>>> for i in range(3): _ = tri_pt_data.InsertNextValue(i)
>>> _ = pt_data.SetScalars(tri_pt_data)

To the cells as well, or instead, if we want.  Don't forget there is
only one cell.

>>> tri_cell_data = vtk.vtkFloatArray()
>>> _ = tri_cell_data.InsertNextValue(3)
>>> _ = cell_data.SetScalars(tri_cell_data)

You can set different types of data into the same dataset:

>>> tri_pt_vecs = vtk.vtkFloatArray()
>>> tri_pt_vecs.SetNumberOfComponents(3)
>>> tri_pt_vecs.InsertNextTuple3(1, 1, 1)
>>> tri_pt_vecs.InsertNextTuple3(2, 2, 2)
>>> tri_pt_vecs.InsertNextTuple3(3, 3, 3)
>>> _ = pt_data.SetVectors(tri_pt_vecs)

If you want to look at what you have, run this code

::

   # ..testcode:: when live
   # make a dataset mapper and actor for our unstructured grid
   mapper = vtk.vtkDataSetMapper()
   mapper.SetInput(ugrid)
   actor = vtk.vtkActor()
   actor.SetMapper(mapper)
   # Create the usual rendering stuff.
   ren = vtk.vtkRenderer()
   renWin = vtk.vtkRenderWindow()
   renWin.AddRenderer(ren)
   iren = vtk.vtkRenderWindowInteractor()
   iren.SetRenderWindow(renWin)
   # add the actor
   ren.AddActor(actor)
   # Render the scene and start interaction.
   iren.Initialize()
   renWin.Render()
   iren.Start()

.. [VTK4]
   Schroeder, Will, Ken Martin, and Bill Lorensen. (2006) *The 
   Visualization Toolkit--An Object-Oriented Approach To 3D Graphics*. : 
   Kitware, Inc.


