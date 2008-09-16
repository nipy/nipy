.. _todo:

=========================
TODO for nipy development
=========================

.. contents::

This document will serve to organize current development work on nipy.
It will include current sprint items, future feature ideas and sprint
items, and design discussions.  This should contain more details than
the :ref:`roadmap`.

Documentation
=============

- There have been many changes to nipy_, the documentation here, on
  Launchpad and the TRAC site should be organized and updated.
 
- Create NIPY sidebar with links to all project related websites.

- Create a Best Practices document.

- Update the README and INSTALL on the Trac Wiki.  These should
  reference a reST formatted version committed to the repository.
  Include information on downloading the fmri data and running the
  tests.

- Add list of dependencies in the INSTALL.

- Post sphinx generated docs to update neuroimaging.scipy.org.

- Document nipy development process with pynifti.  Use of git branches
  and updating the source in nipy (update_source.py).

Bugs
====

These should be moved to the nipy_ bug section on launchpad.  Many
were added here this summer, grouping until they can be input.

- Resolve differences between running tests via nose on command line
  and ni.test().

  ::
  
    cburns@nipy 11:32:32 $ nosetests -sv 
    Ran 216 tests in 135.606s
    FAILED (SKIP=35, errors=3)
    
    In [2]: ni.test()
    Ran 146 tests in 10.874s
    FAILED (SKIP=18, errors=12)
    Out[2]: <nose.result.TextTestResult run=146 errors=12 failures=0>

- Remove creation of named temporary files "\*.nii", use NamedTemporaryFile 
  instead in test modules:

  * modalities/fmri/tests/test_regression.py 
  * modalities/fmri/fmristat/tests/test_model.py

- Cleanup and standardize the axis names and pynifti orientation
  codes.  See failing test in test_axis:test_Axis.test_init,
  presumably the Axis initializer use to check for a valid name before
  assigning.  It now blindly assigns the name.

- Fix test errors for concatenation and replication of sampling grids.
  See test_grid.py.

- Fix .mat file IO.  See test_mapping.py

- Verify documententation of the image generators. Create a simple
  example using them.

- Add test data where volumes contain intensity ramps.  Slice with
  generator and test ramp values.

- Use python 2.5 feature of being able to reset the generator?

- Fix deprecation error in pynifti's swig generated extension code::

    /Users/cburns/src/nipy-trunk/neuroimaging/externals/pynifti/nifti/niftiformat.py:458
    DeprecationWarning: PyArray_FromDims: use PyArray_SimpleNew.  return
    nifticlib.mat442array(self.__nimg.sto_xyz)
    ...
    /Users/cburns/src/nipy-trunk/neuroimaging/externals/pynifti/nifti/niftiformat.py:458
    DeprecationWarning: PyArray_FromDimsAndDataAndDescr: use
    PyArray_NewFromDescr.  return
    nifticlib.mat442array(self.__nimg.sto_xyz)

- Nifti image saving bug.  PixDims not being saved correctly.

Data
====

- Replace fmri test file :file:`funcfile` with a reasonable fmri file.  It's
  shape is odd, (20,20,2,20).  Many tests have been updated to this
  file and will need to me modified.  :file:`funcfile` is located in
  neuroimaging/testing/functinal.nii.gz


Refactorings
============

- Remove path.py and replace datasource with numpy's version.
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

- Rewrite weave code in algorithms/statistics/intrinsic_volumes.py as
  C extension.

- Determine need for odict.py.  Verify origin and license if we
  determine we need it.

- Cleanup neuroimaging.testing directory.  Possibly rename 'testing'
  to 'tests'.  Move utils.tests.data.__init__.py to tests and update
  import statements in all test modules.

- Remove neuroimaging.utils dir. (path.py and odict.py should be in
  externals)

- image.save function should accept filename or file-like object.  If
  I have an open file I would like to be able to pass that in also,
  instead of fp.name.  Happens in test code a lot.

- image._open function should accept Image objects in addition to
  ndarrays and filenames.  Currently the save function has to call
  np.asarray(img) to get the data array out of the image and pass them
  to _open in order to create the output image.

- Add dtype options when saving. When saving images it uses the native
  dtype for the system.  Should be able to specify this.  in the
  test_file_roundtrip, self.img is a uint8, but is saved to tmpfile as
  float64.  Adding this would allow us to save images without the
  scaling being applied.

- In image._open(url, ...), should we test if the "url" is a PyNiftiIO
  object already? This was in the tests from 'old code' and passed::
  
    new = Image(self.img._data, self.img.grid) 

  img._data is a PyNIftiIO object.  It works, but we should verify
  it's harmless otherwise prevent it from happening.

- Rename SamplingGrid to CoordinateMap.  Image.grid to Image.coordmap?

Code Design Thoughts
====================

A central location to dump thoughts that could be shared by the
developers and tracked easily.





Affine
------
- calling affine with load, ImageInterpolate, etc., results in a one-pixel offset
  in the translation columns (x, y and z) of the affine matrix and is related to
  converting python to matlab format.


Imagelist
---------
- remove concatenating grid (composite the mappings?)
- look at Mergeimage function and understand it.
- consider preventing Image from opening 4D. simplfy the user API for 3D/4D.
  create factory function to do this.




Modalities
----------

- Fix fmri.pca module.  Internally it's referencing old image api that
  no longer exists like Image.slice_iterator.  Currently all tests are
  skipped or commented out.

- FmriStat has undefined objects, FmriStatOLS and FmriStatAR.  Look
  into modalities.fmri.fmristat.tests.test_utils.py

- Automated test for pca, check for covariance diagonal structure, post pca.

- Create working example out of this TRAC `pca
  <http://neuroimaging.scipy.org/neuroimaging/ni/wiki/PrincipalComponents>`_
  page.  Should also be a rest document.
  
  

fixes.scipy.ndimage
-------------------

Fix possible precision error in test_registration function
test_autoalign_nmi_value_2.  See FIXME.

Fix error in test_segment test_texture2 function.  See FIXME.

Future Features
---------------

Egg support.  Look to revno 1642, a setup_egg.py that Gael had added.
This was removed as it did not work.  It did appear to allow this
development install option, which we should restore when eggs are working::

    sudo python setup_egg.py develop --prefix /usr/local

Add Fernando's nose fix for running doctests in extension code.  May
get this through numpy?  Fernando was considering adding this there.

Place nipy-io-overhaul up on lp/cburns for matthew reference.

Move header_utils, utils, analyze_to_nifti and sliceplot to
sandbox/tools.  Files are currently in
nipy-sandbox/neuroimaging/data_io/formats.

import neuroimaging.algorithms is very slow!  Find and fix.  The
shared library is slow.

Auto backup script for nipy repos to run as weekly cron job.  Chris
will run this on his machine.

Update import statements to match scipy/numpy standards::

  import numpy as np

Get nifticlib to support bz2.

Questions
---------

- Should millimeter coordinates be expressed in xyz or zyx order?

.. _nipy: https://launchpad.net/nipy
