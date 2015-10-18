.. _release-guide:

***********************************
A guide to making a nipy release
***********************************

A guide for developers who are doing a nipy release

.. _release-tools:

Release tools
=============

There are some release utilities that come with nibabel_.  nibabel should
install these as the ``nisext`` package, and the testing stuff is understandably
in the ``testers`` module of that package.  nipy has Makefile targets for their
use.  The relevant targets are::

    make check-version-info

This installs the code from a git archive, from the repository, and for
in-place use, and runs the ``get_info()`` function to confirm that installation
is working and information parameters are set correctly.

.. _release-checklist:

Release checklist
=================

* Review the open list of `nipy issues`_.  Check whether there are
  outstanding issues that can be closed, and whether there are any issues that
  should delay the release.  Label them !

* Review and update the release notes.  Review and update the :file:`Changelog`
  file.  Get a partial list of contributors with something like::

      git log 0.2.0.. | grep '^Author' | cut -d' ' -f 2- | sort | uniq

  where ``0.2.0`` was the last release tag name.

  Then manually go over ``git shortlog 0.2.0..`` to make sure the release notes
  are as complete as possible and that every contributor was recognized.

* Use the opportunity to update the ``.mailmap`` file if there are any duplicate
  authors listed from ``git shortlog -ns``.

* Add any new authors to the ``AUTHORS`` file.  Add any new entries to the
  ``THANKS`` file.

* Check the copyright years in ``doc/conf.py`` and ``LICENSE``

* Refresh the ``README.rst`` text from the ``LONG_DESCRIPTION`` in ``info.py``
  by running ``make refresh-readme``.

  Check the output of::

    rst2html.py README.rst > ~/tmp/readme.html

  because this will be the output used by pypi_

* Check the dependencies listed in ``nipy/info.py`` (e.g.
  ``NUMPY_MIN_VERSION``) and in ``doc/users/installation.rst``.  They should
  at least match. Do they still hold?  Make sure ``.travis.yml`` is testing
  these minimum dependencies specifically.

* Check the examples in python 2 and python 3, by running something like::

    cd ..
    ./nipy/tools/run_log_examples.py nipy/examples --log-path=~/tmp/eg_logs

  in a python 2 and python 3 virtualenv.  Review the output in (e.g.)
  ``~/tmp/eg_logs``. The output file ``summary.txt`` will have the pass file
  printout that the ``run_log_examples.py`` script puts onto stdout while
  running.

* Do a final check on the `nipy buildbot`_

* If you have travis-ci_ building set up on your own fork of nipy you might
  want to push the code in its current state to a branch that will build,
  e.g::

    git branch -D pre-release-test # in case branch already exists
    git co -b pre-release-test
    git push your-github-user pre-release-test

* Make sure all the ``.c`` generated files are up to date with Cython sources
  with::

    ./tools/nicythize

Release checking - buildbots
============================

* Check all the buildbots pass
* Run the builder and review the possibly green output from
  http://nipy.bic.berkeley.edu/builders/nipy-release-checks

  This runs all of::

    make distclean
    python -m compileall .
    make sdist-tests
    make check-version-info
    make check-files

* You need to review the outputs for errors; at the moment this buildbot builder
  does not check whether these tests passed or failed.
* ``make check-version-info`` checks how the commit hash is stored in the
  installed files.  You should see something like this::

    {'sys_version': '2.6.6 (r266:84374, Aug 31 2010, 11:00:51) \n[GCC 4.0.1 (Apple Inc. build 5493)]', 'commit_source': 'archive substitution', 'np_version': '1.5.0', 'commit_hash': '25b4125', 'pkg_path': '/var/folders/jg/jgfZ12ZXHwGSFKD85xLpLk+++TI/-Tmp-/tmpGPiD3E/pylib/nipy', 'sys_executable': '/Library/Frameworks/Python.framework/Versions/2.6/Resources/Python.app/Contents/MacOS/Python', 'sys_platform': 'darwin'}
    /var/folders/jg/jgfZ12ZXHwGSFKD85xLpLk+++TI/-Tmp-/tmpGPiD3E/pylib/nipy/__init__.pyc
    {'sys_version': '2.6.6 (r266:84374, Aug 31 2010, 11:00:51) \n[GCC 4.0.1 (Apple Inc. build 5493)]', 'commit_source': 'installation', 'np_version': '1.5.0', 'commit_hash': '25b4125', 'pkg_path': '/var/folders/jg/jgfZ12ZXHwGSFKD85xLpLk+++TI/-Tmp-/tmpGPiD3E/pylib/nipy', 'sys_executable': '/Library/Frameworks/Python.framework/Versions/2.6/Resources/Python.app/Contents/MacOS/Python', 'sys_platform': 'darwin'}
    /Users/mb312/dev_trees/nipy/nipy/__init__.pyc
    {'sys_version': '2.6.6 (r266:84374, Aug 31 2010, 11:00:51) \n[GCC 4.0.1 (Apple Inc. build 5493)]', 'commit_source': 'repository', 'np_version': '1.5.0', 'commit_hash': '25b4125', 'pkg_path': '/Users/mb312/dev_trees/nipy/nipy', 'sys_executable': '/Library/Frameworks/Python.framework/Versions/2.6/Resources/Python.app/Contents/MacOS/Python', 'sys_platform': 'darwin'}

* ``make check-files`` checks if the source distribution is picking up all the
  library and script files.  Look for output at the end about missed files, such
  as::

    Missed script files:  /Users/mb312/dev_trees/nipy/bin/nib-dicomfs, /Users/mb312/dev_trees/nipy/bin/nifti1_diagnose.py

  Fix ``setup.py`` to carry across any files that should be in the distribution.
* Check the documentation doctests pass from
  http://nipy.bic.berkeley.edu/builders/nipy-doc-builder

Doing the release
=================

* The release should now be ready.

* Edit :file:`nipy/info.py` to set ``_version_extra`` to ``''``; commit.
  Then::

    make source-release

* Once everything looks good, you are ready to upload the source release to
  PyPi.  See `setuptools intro`_.  Make sure you have a file ``\$HOME/.pypirc``,
  of form::

    [distutils]
    index-servers =
        pypi

    [pypi]
    username:your.pypi.username
    password:your-password

    [server-login]
    username:your.pypi.username
    password:your-password

* Once everything looks good, upload the source release to PyPi.  See
  `setuptools intro`_::

    python setup.py register
    python setup.py sdist --formats=gztar,zip upload

* Trigger binary builds for Windows from the buildbots. See builders
  ``nipy-bdist32-26``, ``nipy-bdist32-27``, ``nipy-bdist32-32``.  The ``exe``
  builds will appear in http://nipy.bic.berkeley.edu/nipy-dist . Download the
  builds and upload to pypi.

* Trigger binary builds for OSX from travis-ci:

    * https://travis-ci.org/MacPython/nipy-wheels
    * https://github.com/MacPython/nipy-wheels

  Upload the resulting wheels to pypi from http://wheels.scipy.org;

* Tag the release with tag of form ``0.5.0``::

    git tag -am 'Second main release' 0.5.0

* Now the version number is OK, push the docs to github pages with::

    make upload-html

* Set up maintenance / development branches

  If this is this is a full release you need to set up two branches, one for
  further substantial development (often called 'trunk') and another for
  maintenance releases.

  * Branch to maintenance::

      git co -b maint/0.5.x

    Set ``_version_extra`` back to ``.dev`` and bump ``_version_micro`` by 1.
    Thus the maintenance series will have version numbers like - say - '0.5.1.dev'
    until the next maintenance release - say '0.5.1'.  Commit. Don't forget to
    push upstream with something like::

      git push upstream maint/0.2.x --set-upstream

  * Start next development series::

      git co main-master

    then restore ``.dev`` to ``_version_extra``, and bump ``_version_minor`` by 1.
    Thus the development series ('trunk') will have a version number here of
    '0.3.0.dev' and the next full release will be '0.3.0'.

  * Merge ``-s ours`` the version number changes from the maint release, e.g::

      git merge -s ours maint/0.3.x

    This marks the version number changes commit as merged, so we can merge any
    changes we need from the maintenance branch without merge conflicts.

  If this is just a maintenance release from ``maint/0.2.x`` or similar, just
  tag and set the version number to - say - ``0.2.1.dev``.

* Push tags::

    git push --tags

* Announce to the mailing lists.

.. _setuptools intro: http://packages.python.org/an_example_pypi_project/setuptools.html
.. _travis-ci: http://travis-ci.org

.. include:: ../../links_names.txt
