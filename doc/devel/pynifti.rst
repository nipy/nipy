.. _pynifti:

============
 Pynifti IO
============

We are using pynifti_ for the underlying nifti file I/O support.
Pynifti is built upon the nifticlibs_ so this will provide us will
*full* nifti support.  We are working closely with the author of
pynifti_, Michael Hanke, and pushing any bug fixes or feature
improvements upstream to the git repository.

Developers should read through Michael's documentation on the pynifti_
site for some details on the project.  The source is checked out from
the `git repository.
<http://git.debian.org/?p=pkg-exppsy/pynifti.git>`_

Using the command::

  git clone http://git.debian.org/git/pkg-exppsy/pynifti.git

Git
---

Pynifti uses git_ for it's version control system.  Git is very
different from `svn <http://subversion.tigris.org/>`_, developers
should read some documentation on git before doing any work with the
git repository.

The git_ website has several tutorials and full documentation.  A good
starting point may be the `Git for SVN Users
<http://git.or.cz/course/svn.html>`_

Git has a unique mechanism for storing multiple branches on your
machine.  Instead of having separate file directories, git will store
all branches in one directory and store *branch diffs* in an internal
database.  When you switch branches (``checkout`` in git terminology),
git will apply the branch diff to the directory, updating any files to
match the new branch.

Development Cycle
-----------------

There are 3 development branches that the nipy developers need to
interact with in the git repository:

* master - the main pynifti repository

* cb/master - Chris' pynifti developer repository

* cb/nipy - Chris' pynifti developer repository with nipy specific code

A nipy development path would look like this:

#. In the master branch, merge with the server to get any updates from
 other pynifti developers.

#. Checkout the cb/master branch and merge from the local master branch.

#. Make any code edits that should be pushed upstream in the cb/master
 branch.  Michael cherry-picks changes into the master branch.

#. Checkout the cb/nipy branch and merge from the cb/master.  The
 cb/nipy branch is used as the source for nipy.

To update the nipy source:

#. Change to the pynifti directory in the nipy developer trunk::

    cd trunk-dev/neuroimaging/externals/pynifti/utils

#. Run the ``update_source.py`` script to update the source.  This
 assumes a directory structure the pynifti sources are in the
 directory ``$HOME/src/pynifti``.  Run the script::

    python update_source.py


.. _git: http://git.or.cz/
.. _pynifti: http://niftilib.sourceforge.net/
.. _nifticlibs: http://nifti.nimh.nih.gov/
