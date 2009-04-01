.. _todo:

===========================
 TODO for nipy development
===========================

This document will serve to organize current development work on nipy.
It will include current sprint items, future feature ideas, and design
discussions, etc...

Current Sprint
==============

**Goal for Feb 27**

Cleanup and document the Image and CoordinateMap classes.  There have
been various changed to the CoordinateMap classes lately, merge
Jonathan's branch, finish remaining changes, update docstrings and
doctests.  Write tutorials explaining these base classes.

Working prototype for interfacing with SPM.

Working prototype for registration visualization.

**Goal for March 20**

Review fff2.neuro code and prepare for sprint.

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

These should be moved to the nipy_ bug section on launchpad.  Placed
here until they can be input.


* Fix fmri.pca module.  Internally it's referencing old image api that
  no longer exists like Image.slice_iterator.  Currently all tests are
  skipped or commented out.

* FmriStat has undefined objects, FmriStatOLS and FmriStatAR.  Look
  into modalities.fmri.fmristat.tests.test_utils.py

* Fix possible precision error in
  fixes.scipy.ndimage.test_registration function
  test_autoalign_nmi_value_2.  See FIXME.

* Fix error in test_segment test_texture2 functions
  (fixes.scipy.ndimage).  See FIXME.

* import neuroimaging.algorithms is very slow!  Find and fix.  The
  shared library is slow.

* base class for all new-style classes should be *object*

Data
====

* Replace fmri test file :file:`funcfile` with a reasonable fmri file.  It's
  shape is odd, (20,20,2,20).  Many tests have been updated to this
  file and will need to me modified.  :file:`funcfile` is located in
  neuroimaging/testing/functinal.nii.gz


Refactorings
============

* Remove path.py and replace datasource with numpy's version.
  datasource and path.py cleanup should be done together as nipy's
  datasource is one of the main users of path.py:

  * Convert from nipy datasource to numpy datasource.  Then remove
    nipy's datasource.py

  * Delete neuroimaging/utils/path.py.  This custom path module does
    not provide any benefit over os.path.  Using a non-standard path
    module adds confusion to the code.  This will require going
    through the code base and updating all references to the path
    module.  Perhaps a good use of grin for a global search and
    replace.

* Rewrite weave code in algorithms/statistics/intrinsic_volumes.py as
  C extension.

* Cleanup neuroimaging.testing directory.  Possibly rename 'testing'
  to 'tests'.  Move utils.tests.data.__init__.py to tests and update
  import statements in all test modules.

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

* Automated test for modalities.fmri.pca, check for covariance
  diagonal structure, post pca.

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
