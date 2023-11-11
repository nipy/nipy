.. _release-guide:

***********************************
A guide to making a nipy release
***********************************

A guide for developers who are doing a nipy release

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

  Then manually go over ``git shortlog 0.2.0..`` to make sure the release
  notes are as complete as possible and that every contributor was recognized.

* Use the opportunity to update the ``.mailmap`` file if there are any
  duplicate authors listed from ``git shortlog -ns``.

* Add any new authors to the ``AUTHORS`` file.  Add any new entries to the
  ``THANKS`` file.

* Check the copyright years in ``doc/conf.py`` and ``LICENSE``

* Check the output of::

    rst2html.py README.rst > ~/tmp/readme.html

  because this will be the output used by PyPI_

* Check the dependencies listed in ``pyproject.toml`` and in
  ``requirements.txt`` and in ``doc/users/installation.rst``.  They should at
  least match. Do they still hold?  Make sure ``.travis.yml`` is testing these
  minimum dependencies specifically.

*   Check the examples in python 2 and python 3, by
    running something like::

        cd ..
        ./nipy/tools/run_log_examples.py nipy/examples --log-path=~/tmp/eg_logs

    in a Python 2 and python 3 virtualenv.  Review the output in (e.g.)
    ``~/tmp/eg_logs``. The output file ``summary.txt`` will have the pass file
    printout that the ``run_log_examples.py`` script puts onto stdout while
    running.

    You might want to do a by-eye comparison between the 2.7 and 3.x files
    with::

        diff -r nipy-examples-2.7 nipy-examples-3.5 | less

* If you have travis-ci_ building set up on your own fork
  of Nipy you might want to push the code in its current
  state to a branch that will build, e.g::

    git branch -D pre-release-test # in case branch already exists
    git co -b pre-release-test
    git push your-github-user pre-release-test

* Make sure all the ``.c`` generated files are up to date
  with Cython sources with::

    ./tools/nicythize

* Check the documentation doctests pass::

    virtualenv venv
    venv/bin/activate
    pip install -r doc-requirements.txt
    pip install -e .
    (cd docs && make clean-doctest)

* Check the doc build::

    virtualenv venv
    venv/bin/activate
    pip install -r doc-requirements.txt
    pip install -e .
    (cd docs && make html)

* Build and test the Nipy wheels.  See the `wheel builder README
  <https://github.com/MacPython/nipy-wheels>`_ for instructions.  In summary,
  clone the wheel-building repo, edit the ``.travis.yml`` and ``appveyor.yml``
  text files (if present) with the branch or commit for the release, commit
  and then push back up to github.  This will trigger a wheel build and test
  on OSX, Linux and Windows. Check the build has passed on on the Travis-CI
  interface at https://travis-ci.org/MacPython/nipy-wheels.  You'll need
  commit privileges to the ``nipy-wheels`` repo; ask Matthew Brett or on the
  mailing list if you do not have them.

Doing the release
=================

* The release should now be ready.

* Edit :file:`nipy/__init__.py` to set ``version`` to e.g. ``0.5.2``; commit.
  Then::

    make source-release

* For the wheel build / upload, follow the `wheel builder README`_
  instructions again.  Push.  Check the build has passed on the Github
  interface.  Now follow the instructions in the page above to download the
  built wheels to a local machine and upload to PyPI.

* Once everything looks good, you are ready to upload the source release to
  PyPI.  See `setuptools intro`_.  Make sure you have a file ``\$HOME/.pypirc``,
  of form::

    [distutils]
    index-servers =
        pypi

    [pypi]
    username:your.pypi.username
    password:your-password

* Sign and upload the source release to PyPI using Twine_::

    gpg --detach-sign -a dist/nipy*.tar.gz
    twine upload dist/nipy*.tar.gz*

* Tag the release with tag of form ``0.5.0``. `-s` below makes a signed tag::

    git tag -s 'Second main release' 0.5.0

* Now the version number is OK, push the docs to github pages with::

    make upload-html

*   Set up maintenance / development branches

    If this is this is a full release you need to set up two branches, one for
    further substantial development (often called 'trunk') and another for
    maintenance releases.

    *   Start next development series::

            git co main-master

        then restore ``.dev`` to ``__version__``, and bump
        the minor version by 1. Thus the development series ('trunk') will
        have a version number here of '0.3.0.dev' and the next full release
        will be '0.3.0'.

    *   Merge ``-s ours`` the version number changes from the maint release,
        e.g::

          git merge -s ours maint/0.3.x

        This marks the version number changes commit as merged, so we can
        merge any changes we need from the maintenance branch without merge
        conflicts.

    If this is just a maintenance release from ``maint/0.2.x`` or similar, just
    tag and set the version number to - say - ``0.2.1.dev``.

* Push tags::

    git push --tags

* Announce to the mailing lists.

.. _setuptools intro: http://packages.python.org/an_example_pypi_project/setuptools.html
.. _twine: https://pypi.python.org/pypi/twine
.. _travis-ci: http://travis-ci.org

.. include:: ../../links_names.txt
