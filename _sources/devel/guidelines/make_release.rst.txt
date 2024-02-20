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

      PREV_RELEASE=0.5.0
      git log $PREV_RELEASE.. | grep '^Author' | cut -d' ' -f 2- | sort | uniq

  where ``0.5.0`` was the last release tag name.

  Then manually go over ``git shortlog $PREV_RELEASE..`` to make sure the
  release notes are as complete as possible and that every contributor was
  recognized.

* Use the opportunity to update the ``.mailmap`` file if there are any
  duplicate authors listed from ``git shortlog -ns``.

* Add any new authors to the ``AUTHOR`` file.  Add any new entries to the
  ``THANKS`` file.

* Check the copyright years in ``doc/conf.py`` and ``LICENSE``

* Check the output of::

    rst2html.py README.rst > ~/tmp/readme.html

  because this will be the output used by PyPI_

* Check the dependencies listed in ``pyproject.toml`` and in
  ``requirements.txt`` and in ``doc/users/installation.rst``.  They should at
  least match. Do they still hold?  Make sure ``.github/workflows`` is testing
  these minimum dependencies specifically.

*   Check the examples.  First download the example data
    by running something like::

        # Install data packages.
        pip install https://nipy.org/data-packages/nipy-templates-0.3.tar.gz
        pip install https://nipy.org/data-packages/nipy-data-0.3.tar.gz

    Then run the tests on the examples with::

        # Move out of the source directory.
        cd ..
        # Make log file directory.
        mkdir ~/tmp/eg_logs
        ./nipy/tools/run_log_examples.py nipy/examples --log-path=~/tmp/eg_logs

    in a virtualenv.  Review the output in (e.g.) ``~/tmp/eg_logs``. The
    output file ``summary.txt`` will have the pass file printout that the
    ``run_log_examples.py`` script puts onto stdout while running.

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
  clone the wheel-building repo, edit the ``.github/workflow`` text files (if
  present) with the branch or commit for the release, commit and then push
  back up to github.  This will trigger a wheel build and test on macOS, Linux
  and Windows. Check the build has passed on on the Github interface at
  https://travis-ci.org/MacPython/nipy-wheels.  You'll need commit privileges
  to the ``nipy-wheels`` repo; ask Matthew Brett or on the mailing list if you
  do not have them.

Doing the release
=================

* The release should now be ready.

* Edit :file:`nipy/__init__.py` to set ``__version__`` to e.g. ``0.6.0``.

  Edit :file:`meson.build` to set ``version`` to match.

  Commit, then::

    make source-release

* For the wheel build / upload, follow the `wheel builder README`_
  instructions again.  Push.  Check the build has passed on the Github
  interface.  Now follow the instructions in the page above to download the
  built wheels to a local machine and upload to PyPI.

* Once everything looks good, you are ready to upload the source release to
  PyPI.  See `setuptools intro`_.  Make sure you have a file ``\$HOME/.pypirc``,
  of form::

    [pypi]
    username = __token__

* Sign and upload the source release to PyPI using Twine_::

    gpg --detach-sign -a dist/nipy*.tar.gz
    twine upload dist/nipy*.tar.gz*

* Tag the release with tag of form ``0.6.0``. `-s` below makes a signed tag::

    git tag -s 'Second main release' 0.6.0

* Now the version number is OK, push the docs to github pages with::

    make upload-html

*   Start the new series.

    Edit ``nipy/__init__.py`` and set version number to something of form::

        __version__ = "0.6.1.dev1"

    where ``0.6.0`` was the previous release.

* Push tags::

    git push --tags

* Announce to the mailing lists.

.. _twine: https://pypi.python.org/pypi/twine

.. include:: ../../links_names.txt
