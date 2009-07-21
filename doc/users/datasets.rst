
=============================
Volumetrique data structures
=============================

Volumetric data structures expose numerical values embedded in a world
space. For instance, a volume could expose the T1 intensity, as acquired
in scanner space, or the BOLD signal in MNI152 template space. The values
can be multi-dimensional, in the case of a BOLD signal, the fMRI signal
would correspond to a time series at each position in world space.

.. currentmodule:: nipy.datasets.volumes.volume_img

The image structure: :class:`VolumeImg`
=======================================

The structure most often used in neuroimaging is the :class:`VolumeImg`.
It corresponds, for instance, to the structure used in the Nifti files.

We call a volume-image (class: :class:`VolumeImg`) a volumetric
datastructure given by data points lying on a regular grid: this
structure is a generalization of an image in 3D. The voxels, vertices of
the grid, are mapped to coordinnates by an affine transformation. As a
result, the grid is regular and evenly-spaced, but may not be
orthogonal, and the spacing may differ in the 3 directions.

.. image:: datasets/volume_img.jpg

The data is exposed in a multi dimensional array, with the 3 first axis
corresponding to spatial directions. A complete description of this
object can be found on the page: :class:`VolumeImg`.


More general data structures
===============================

The :class:`VolumeImg` is the most commonly found data structure, and the
simplest to understand, however, volumetric data can be described in more
generic terms, and for performance reason it might be interesting to use
other objects.

Here, we give a list of the nipy volumetric data structures, from most
specific, to most general.

Implemented classes
--------------------

Implemented classes (or `concrete` classes) are structures that you can
readily use directly from nipy.

.. currentmodule:: nipy.datasets.volumes.volume_grid

:class:`VolumeGrid`
  In a :class:`VolumeGrid`, the data points are sampled on a 3D grid, but
  unlike for a :class:`VolumeImg`, grid may not be regular. For instance,
  it can be a grid that has been warped by a non-affine transformation.
  Like with the :class:`VolumeImg`, the data is exposed in a multi
  dimensional array, with the 3 first axis corresponding to spatial
  directions.

  .. image:: datasets/volume_grid.jpg

  .. currentmodule:: nipy.datasets.volumes.volume_data

Abstract classes
------------------

Abstract classes cannot be used because they are incompletely
implemented. They serve as to define the interface: the type of objects
that you can use, or how you can extend nipy by exposing the same
set of methods and attributes (the `interface`).

:class:`VolumeData`
  In this volumetric structure, the data is sampled for some points in
  the world space. The object knows how to interpolate between these
  points. The underlying values are stored in a multidimensional array-like 
  object that can be indexed and sliced.

  .. image:: datasets/volume_data.jpg

  This is an abstract base class: it defines an interface, but is not
  fully functional, and can be used only via its children class (such as
  :class:`VolumeGrid` or :class:`VolumeImg`).

  .. currentmodule:: nipy.datasets.volumes.volume_field


:class:`VolumeField`
  This is the most general volumetric structure (base class): all the
  nipy volume expose this interface. This structure does not make any
  assumptions on how the values are internal represented, they may, for
  instance, be represented as a function, rather than as data points, or
  as a data structure that is not an array, such as a graph.

  .. image:: datasets/volume_field.jpg

  This is also an abstract base class: it defines the core nipy
  volumetric data structure interface: you can rely on all the methods
  documented for this class in any nipy data structure.


