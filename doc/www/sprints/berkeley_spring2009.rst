.. _berkeley_spring2009:

=====================
 Berkeley March 2009
=====================

We had a couple springs in the Spring of 2009 at UC Berkeley, since
there is overlap on the content discussed and working one, I've
grouped all the information into one sprint doc.


Venue
=====

The sprint was held at the `Brain Imaging Center`_ at UC Berkeley, in
10 Gianinni Hall.


Code Merge
==========

One of the primary goals of the sprint was to integrate the
neurospin/fff codebase into nipy.  At the 2008 March Paris sprint, a
first start was made on this, by removing the GPL dependency from fff
and looking into Cython and the current Numpy APIs.  The fff code has
been merged into nipy and now lives in nipy.neurospin.  Work will
continue on better integrating the code over the next year.


Image Class
===========

There was a lot of discussion on the existing image class and how to
redesign it in order to make it easier to use, yet provide the
necessary functionality.

Taking into account the discussions at the sprint and some discussion
with Jonathan afterwards, this is the current plan for the Image
Class:

- Images are 3 dimensions or more, always

- First dimensions of axes follow nifti convention of *xyzt* ordering:

  - Axes 0 is x axis
  - Axes 1 is the y axis
  - Axes 2 is the z axis
  - Axes 3 is the time axis

- The image.affine attribute will always be a 4x4 ndarray.

- Coordmap will exist but not be a "public" part of an image object.
  One could always get it through a get_coordmap() method.  It's
  agreed that CoordinateMaps are necessary for non-linear registation
  (and any other non-linear mapping).  But users will rarely, if ever,
  have to deal with them directly.

- The image.data attribute will continue to be a property, scaling the
  data on access.  There will be a clean interface to the raw,
  unscaled data but this will not be the preferred means of accessing
  the data.


Class hierarchy
---------------

We agreed to keep our class hierarchy shallow so that we don't end up
with a large inheritance chain.  We will have one base image class
that all other images inherit from.  The second layer of classes will
be modality specific.

The base image class will have these attrs:
- data : a ndarray
- affine : a 4x4 ndarray
- coordsys : Need to determine if this is a string or a CoordinateSystem

Constructed with (or other utility functions like `nipy.load`)::
  
  img = Image(data, affine, coordsys)

where coordsys can be a user defined string of coordinate axes names
`'xyz'`, or one of a small set of predefined nipy standard coordinate
systems `nipy.mni`.

Class hierarchy::

  class Image(object):
    def __init__(data, affine, coordsys):
      self.data = data
      self.affine = affine
      self.coordsys = coordsys

  class FmriImage(Image):
    # Extend image class, adding FMRI specific attributes and methods
    ...
    self.TR = tr

  class DtiImage(Image):
    # Extend image class, adding DTI specific attributes and methods


Image IO
========

Michael Hanke and Matthew Brett began merging Matthew's pure-python
image IO code, *volumeimages* into pynifti_.  They plan to replace the
nifticlib with volumeimages, but verify the code against the
nifticlib.

A question came up regarding how to know when an image has been
modified and needs resaving before further processing.  The user could
modify either the header, the affine or the data and we need to be
aware of this.

For example, if the user loads the image, then modifies the header,
that image needs to be resaved before being passed to fsl::

  img = load('foo.nii')
  # update header in some way
  img.header['slice_duration'] = 0.200
  realigner = nipy.interfaces.fsl.flirt()
  realigner.run(img, img2)

To deal with these situations, we decided the base image class would
have a *dirty bit*, actually a dirty attr that will be a boolean flag
specifying whether the images is dirty (has been modified) in any way.

Interfaces (or any code that needs to verify the data is clean before
processing) will call a utility function that handles all the logic of
checking the dirty bit and resaving the image if necessary.  This
utility function will return a valid filename.

::

  filename = ni.get_clean_file(img)

`get_clean_file` will accept a nipy Image object or a filename.  

#. If the parameter is an Image object it will reload the header off
   disk and compare the original header with the header in img.  If
   they differ, save and return a new file.  Similarly it will check
   the affine.  If the user accesses the data in _any_way_, even to
   simple view it, the dirty bit will be set, forcing a resave.  This
   constraint on the data access is to ensure the 'right thing'
   happens without burdening the user with having to manually set the
   dirty bit themselves, or make sure they resave the data before
   further processing.

#. If the parameter is an Image object but has never been saved to
   disk, it will be saved and the filename returned.

#. If the parameter is a string, `get_clean_file` will verify it's a
   valid filename and return it.


Interfaces and Pipelines
========================

Satra, Cindee and Dav joined efforts to work on interfaces to FSL_ and
SPM_ and a pipelining API.

Time Series
===========

Ariel, Mike and Paul worked on the time-series branch.

Statistics
==========

Jonathan has updated the stats code to use sympy_ for specifying the
terms and generating the design matrix for the analysis.

Debian Packaging
================

Michael and Yaroslav created a debian package of nipy.  There is still
some work to do on this front, but it was a great start.

Participants
============

Alexis Roche

Ariel Rokem

Bertrand Thirion

Christopher Burns

Cindee Madison

Dav Clark

Fernando Perez

Gael Varoquaux

Jarrod Millman

JB Poline

Jonathan Taylor

Matthew Brett

Michael Hanke

Mike Trumpis

Paul Ivanov

Satrajit Ghosh

Yaroslav Halchenko

.. include:: ../../links_names.txt
