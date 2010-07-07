.. _todo:

===========================
 TODO for nipy development
===========================

This document will serve to organize current development work on nipy.
It will include current sprint items, future feature ideas, and design
discussions, etc...

Documentation
=============

* Create NIPY sidebar with links to all project related websites.
* Create a Best Practices document.
* Create a rst doc for *Request a review* process.

Tutorials
---------

Tutorials are an excellent way to document and test the software.
Some ideas for tutorials to write in our Sphinx documentation (in no
specific order):

* Slice timing
* Image resampling
* Image IO
* Registration using SPM/FSL
* FMRI analysis
* Making one 4D image from many 3D images, and vice versa.  Document
  ImageList and FmriImageList.
* Apply SPM registration .mat to a NIPY image.

* Create working example out of this TRAC `pca
  <http://neuroimaging.scipy.org/neuroimaging/ni/wiki/PrincipalComponents>`_
  page.  Should also be a rest document.

* Add analysis pipeline(s) blueprint.


Bugs
====

These should be moved to the nipy_ bug section on github.  Placed
here until they can be input.

* Fix possible precision error in
  fixes.scipy.ndimage.test_registration function
  test_autoalign_nmi_value_2.  See FIXME.

* Fix error in test_segment test_texture2 functions
  (fixes.scipy.ndimage).  See FIXME.

* import nipy.algorithms is very slow!  Find and fix.  The
  shared library is slow.

* base class for all new-style classes should be *object*; preliminary
  search with ``grin "class +[a-zA-Z0-9]+ *:"``

Refactorings
============

* image.save function should accept filename or file-like object.  If
  I have an open file I would like to be able to pass that in also,
  instead of fp.name.  Happens in test code a lot.

* image._open function should accept Image objects in addition to
  ndarrays and filenames.  Currently the save function has to call
  np.asarray(img) to get the data array out of the image and pass them
  to _open in order to create the output image.

* Add dtype options when saving. When saving images it uses the native
  dtype for the system.  Should be able to specify this.  in the
  test_file_roundtrip, self.img is a uint8, but is saved to tmpfile as
  float64.  Adding this would allow us to save images without the
  scaling being applied.

* In image._open(url, ...), should we test if the "url" is a PyNiftiIO
  object already? This was in the tests from 'old code' and passed::
  
    new = Image(self.img._data, self.img.grid) 

  img._data is a PyNIftiIO object.  It works, but we should verify
  it's harmless otherwise prevent it from happening.

* Look at image.merge_image function.  Is it still needed?  Does it
  fit into the current api?

* FmriImageList.emptycopy() - Is there a better way to do this?
  Matthew proposed possibly implementing Gael's dress/undress metadata
  example.

* Verify documentation of the image generators. Create a simple
  example using them.

* Use python 2.5 feature of being able to reset the generator?

* Add test data where volumes contain intensity ramps.  Slice with
  generator and test ramp values.

* Implement `fmriimagelist blueprint
  <https://blueprints.launchpad.net/nipy/+spec/fmriimagelist>`_.

Code Design Thoughts
====================

A central location to dump thoughts that could be shared by the
developers and tracked easily.

Future Features
===============

Put ideas here for features nipy should have but are not part of our
current development.  These features will eventually be added to a
weekly sprint log.

* Auto backup script for nipy repos to run as weekly cron job.  We
  should setup a machine to perform regular branch builds and tests.
  This would also provide an on-site backup.

* See if we can add bz2 support to nifticlib.

* Should image.load have an optional squeeze keyword to squeeze a 4D
  image with one frame into a 3D image?


.. include:: ../../links_names.txt
