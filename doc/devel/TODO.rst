.. _todo:

=========================
TODO for nipy development
=========================

.. contents::

This document will serve to organize current development work on nipy.
It will include current sprint items, future feature ideas and sprint
items, and design discussions.  This should contain more details than
the :ref:`roadmap`.


Decomissioning Trac
===================

A few pages from Trac we probably want to keep:

* http://projects.scipy.org/neuroimaging/ni/wiki/CodingSprintOne
* http://projects.scipy.org/neuroimaging/ni/wiki/CodingSprintTwo
* http://projects.scipy.org/neuroimaging/ni/wiki/SprintCodeSharing


Documentation
=============

* There have been many changes to nipy_, the documentation here, on
  Launchpad and the TRAC site should be organized and updated.
 
* Create NIPY sidebar with links to all project related websites.

* Create a Best Practices document.

* Update the README and INSTALL on the Trac Wiki.  These should
  reference a reST formatted version committed to the repository.
  Include information on downloading the fmri data and running the
  tests.

* Add list of dependencies in the INSTALL.

* Create working example out of this TRAC `pca
  <http://neuroimaging.scipy.org/neuroimaging/ni/wiki/PrincipalComponents>`_
  page.  Should also be a rest document.

* Add analysis pipeline(s) blueprint.

Bugs
====

These should be moved to the nipy_ bug section on launchpad.  Many
were added here this summer, grouping until they can be input.

* Resolve differences between running tests via nose on command line
  and ni.test().

  ::
  
    cburns@nipy 11:32:32 $ nosetests -sv 
    Ran 216 tests in 135.606s
    FAILED (SKIP=35, errors=3)
    
    In [2]: ni.test()
    Ran 146 tests in 10.874s
    FAILED (SKIP=18, errors=12)
    Out[2]: <nose.result.TextTestResult run=146 errors=12 failures=0>

* Remove creation of named temporary files "\*.nii", use NamedTemporaryFile 
  instead in test modules:

  * modalities/fmri/tests/test_regression.py 
  * modalities/fmri/fmristat/tests/test_model.py

* Cleanup and standardize the axis names and pynifti orientation
  codes.  See failing test in test_axis:test_Axis.test_init,
  presumably the Axis initializer use to check for a valid name before
  assigning.  It now blindly assigns the name.

* Fix test errors for concatenation and replication of sampling grids.
  See test_grid.py.

* Fix .mat file IO.  See test_mapping.py

* Fix deprecation error in pynifti's swig generated extension code::

    /Users/cburns/src/nipy-trunk/neuroimaging/externals/pynifti/nifti/niftiformat.py:458
    DeprecationWarning: PyArray_FromDims: use PyArray_SimpleNew.  return
    nifticlib.mat442array(self.__nimg.sto_xyz)
    ...
    /Users/cburns/src/nipy-trunk/neuroimaging/externals/pynifti/nifti/niftiformat.py:458
    DeprecationWarning: PyArray_FromDimsAndDataAndDescr: use
    PyArray_NewFromDescr.  return
    nifticlib.mat442array(self.__nimg.sto_xyz)

* Nifti image saving does not preserve the header values.  image.save
  looses this information as the PyNiftiIO constructor pulls the data
  array out of the Nipy Image and saves that only.

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

* Determine need for odict.py.  Verify origin and license if we
  determine we need it.

* Cleanup neuroimaging.testing directory.  Possibly rename 'testing'
  to 'tests'.  Move utils.tests.data.__init__.py to tests and update
  import statements in all test modules.

* Remove neuroimaging.utils dir. (path.py and odict.py should be in
  externals)

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

* Consider removing class ConcatenatedGrid in grid.py.  Is this
  functionality provided in the ImageList class?

* Look at image.merge_image function.  Is it still needed?  Does it
  fit into the current api?

* Provide clear documentation and examples for how to use Image,
  ImageList, and FmriImageList classes with 3D and 4D images.  It
  should be clear to the user when to use each and we should provide a
  clean api to move images between them.

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

Code Design Thoughts
====================

A central location to dump thoughts that could be shared by the
developers and tracked easily.

Future Features
===============

Put ideas here for features nipy should have but are not part of our
current development.  These features will eventually be added to a
weekly sprint log.

* Egg support.  Look to revno 1642, a setup_egg.py that Gael had
  added.  This was removed as it did not work.  It did appear to allow
  this development install option, which we should restore when eggs
  are working::

    sudo python setup_egg.py develop --prefix /usr/local

* Create a nipy tools repos that can be shared by the team.  Include
  tools for building like makepkg, tools from the old utils directory,
  header_utils and analyze_to_nifti, etc...

* Auto backup script for nipy repos to run as weekly cron job.  We
  should setup a machine to perform regular branch builds and tests.
  This would also provide an on-site backup.

* See if we can add bz2 support to nifticlib.

Questions
=========

* Should millimeter coordinates be expressed in xyz or zyx order?

  **Answer:** xyz order.

  **Note:** we should probably change the names of the "VoxelAxes" to
    something other than 'x,y,z', at least in creating CoordinateMaps

Weekly Sprint
=============

This will hold our current sprint items and be updated weekly as we
work through the backlog.

**Goal:**

*Fix bugs and implement any functionality needed to begin registration
next week.*


* Implement `fmriimagelist blueprint
  <https://blueprints.launchpad.net/nipy/+spec/fmriimagelist>`_.

  * Requires some changes to CoordinateMap?
  * FmriImageList has a frametimes attr.  Document it and consider
    renaming to volume_start_times.

* CoordinateMap API: Create a blueprint for the public api.  Implement
  any needed functionality for registration.

* Image saving with dtype support.  Handle slope and intercept
  correctly for dtype downcasting.

* Image.affine in xyz millimeter ordering.  Reading is working, need
  to fix writing and add tests for roundtrip.  Test various ijk
  orderings.

* Fix memory error in pynifti when running tests via nosetests. (Only
  happens on Matthew's machine.)

* Work on viewer:
  
  * Review Mike's code.
  * Merge Tom and Chris versions.  Make overlay's work.
    Lightbox/montage viewer if time permitting.

* Should image.load have an optional squeeze keyword to squeeze a 4D
  image with one frame into a 3D image?

* Add *bzr whoami* to bzr_workflow.

* Create a rst doc for *Request a review* process.

.. include:: ../links_names.txt
