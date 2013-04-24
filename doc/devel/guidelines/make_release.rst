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
    make check-files
    make sdist-tests

The first installs the code from a git archive, from the repository, and for
in-place use, and runs the ``get_info()`` function to confirm that installation
is working and information parameters are set correctly.

The second (``sdist-tests``) makes an sdist source distribution archive,
installs it to a temporary directory, and runs the tests of that install.

If you have a version of nibabel trunk past February 11th 2011, there will also
be a functional make target::

    make bdist-egg-tests

This builds an egg (which is a zip file), hatches it (unzips the egg) and runs
the tests from the resulting directory.

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

* Check the examples in python 2 and python 3, by running something like::

    cd ..
    ./nipy/tools/run_log_examples.py nipy/examples --log-path=~/tmp/eg_logs

  in a python 2 and python 3 virtualenv.  Review the output in (e.g.)
  ``~/tmp/eg_logs``. The output file ``summary.txt`` will have the pass file
  printout that the ``run_log_examples.py`` script puts onto stdout while
  running.

* Check the ``long_description`` in ``nipy/info.py``.  Check it matches the
  ``README`` in the root directory, maybe with ``vim`` ``diffthis`` command.

* Do a final check on the `nipy buildbot`_

* If you have travis-ci_ building set up you might want to push the code in its
  current state to a branch that will build, e.g::

    git branch -D pre-release-test # in case branch already exists
    git co -b pre-release-test
    git push origin pre-release-test

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
    make bdist-egg-tests
    make check-version-info
    make check-files

* You need to review the outputs for errors; at the moment this buildbot builder
  does not check whether these tests passed or failed.
* ``make bdist-egg-tests`` may well fail because of a problem with the script
  tests; if you have a recent (>= Jan 15 2013) nibabel ``nisext`` package, you
  could try instead doing::

    python -c 'from nisext.testers import bdist_egg_tests; bdist_egg_tests("nipy", label="not slow and not script_test")'

  Eventually we should update the ``bdist-egg-tests`` makefile target.
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
* You may have virtualenvs for different python versions.  Check the tests
  pass for different configurations.  If you have pytox_ and a network
  connection, and lots of pythons installed, you might be able to do::

    tox

  and get tests for python 2.5, 2.6, 2.7, 3.2.  I (MB) have my own set of
  virtualenvs installed and I've set them up to run with::

    tox -e python25,python26,python27,python32,np-1.2.1

  The trick was only to define these ``testenv`` sections in ``tox.ini``.

  These two above run with::

    make tox-fresh
    make tox-stale

  respectively.

  The long-hand not-tox way looks like this::

    workon python26
    make sdist-tests
    deactivate

  etc for the different virtualenvs.

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

* Trigger binary builds for OSX from the buildbots ``nipy-bdist-mpkg-2.6``,
  ``nipy-bdist-mpkg-2.7``, ``nipy-bdist-mpkg-3.3``. ``egg`` and ``mpkg`` builds
  will appear in http://nipy.bic.berkeley.edu/nipy-dist .  Download the eggs and
  upload to pypi.

* Download the ``mpkg`` builds, maybe with::

    scp -r buildbot@nipy.bic.berkeley.edu:nibotmi/public_html/nipy-dist/*.mpkg .

  Make sure you have `github bdist_mpkg`_ installed, for the root user.  For
  each ``mpkg`` directory, run::

    sudo reown_mpkg nipy-0.3.0.dev-py2.6-macosx10.6.mpkg root admin
    zip -r nipy-0.3.0.dev-py2.6-macosx10.6.mpkg.zip nipy-0.3.0.dev-py2.6-macosx10.6.mpkg

  Upload the ``mpkg.zip`` files. (At the moment, these don't seem to store the
  scripts - needs more work)

* Tag the release with tag of form ``0.3.0``::

    git tag -am 'Second main release' 0.3.0

* Now the version number is OK, push the docs to sourceforge with::

    cd doc
    make upload-stable-web-mysfusername

  where ``mysfusername`` is obviously your own sourceforge username.

* Set up maintenance / development branches

  If this is this is a full release you need to set up two branches, one for
  further substantial development (often called 'trunk') and another for
  maintenance releases.

  * Branch to maintenance::

      git co -b maint/0.2.x

    Set ``_version_extra`` back to ``.dev`` and bump ``_version_micro`` by 1.
    Thus the maintenance series will have version numbers like - say - '0.2.1.dev'
    until the next maintenance release - say '0.2.1'.  Commit. Don't forget to
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

.. _pytox: http://codespeak.net/tox
.. _setuptools intro: http://packages.python.org/an_example_pypi_project/setuptools.html
.. _travis-ci: http://travis-ci.org

.. include:: ../../links_names.txt
