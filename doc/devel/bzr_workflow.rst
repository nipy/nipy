======================
 nipy bazaar workflow
======================

.. Contents::

Overview
--------

This document describes the official workflow for nipy development
using bazaar and the launchpad_ hosting service.  All developers are
expected to adopt this workflow for development.  This workflow is
flexible enough that even an occassional developer can contribute code
to nipy since it does not require all contributors to have commit
access to the main trunk.  This particular workflow was developed to
achieve these primary objectives.

#. Maintain a linear revision number history on the main nipy-trunk.
#. Provide a clear process for code reviews.
#. Minimize damage to the trunk during large code transitions.


Launchpad structure
-------------------
On launchpad_, there exists one main nipy_ trunk that only administrators
have commit access to.  All developers have their own
launchpad.net/~developer directory which may be registered with the
nipy_ project.  In addition, there may be shared team directories on
nipy_, for feature development, but in general individual development
should happen in your ~developer directory.

A snapshot of the nipy_ code directories on launchpad_

* lp:nipy - the nipy-trunk
* ~cburns/nipy - developer branch for cburns (registered with nipy)
* ~twaite/nipy - developer branch for twaite (registered with nipy)

Developer directory structure
-----------------------------
Your development directory will contain two subdirectories:

* nipy-repo/trunk-lp
* nipy-repo/trunk-dev

*trunk-lp* is a pull from the *lp:nipy*.  *trunk-dev* is a branch from
your user directory on launchpad_ and is your working, development
directory. 

Initializing your development environment
----------------------------------------- 

We're using the `shared repository
http://bazaar-vcs.org/SharedRepositoryTutorial`_ feature of bazaar_.

#. Create a *shared repository*::

  bzr init-repo --trees nipy-repo
  cd nipy-repo

#. Checkout the nipy-trunk::

  bzr branch lp:nipy trunk-lp

#. Create a development branch::

  bzr branch trunk-lp trunk-dev

#. Push your development branch up to launchpad_::

  cd trunk-dev
  bzr push bzr+ssh://USER@bazaar.launchpad.net/~USER/nipy/nipy-trunk

Daily development cycle
-----------------------

::

 cd trunk-dev
 # edit code
 bzr ci -m "meaningful message"
 # repeat
 ...
 # periodically push to lp:~user (like end of day)
 bzr push


Weekly/Monthly development practice
-----------------------------------

Semi-regularly merge from `trunk-lp` to `trunk-dev` so that your
working copy is kept in sync with the trunk.  (Also semi-regularly
pull from lp:nipy)

Pull from launchpad to update your trunk copy:::

 cd trunk-lp
 bzr pull lp:nipy

Merge `trunk-lp` into your development directory `trunk-dev`:::

 cd trunk-dev
 bzr merge ../trunk-lp
 bzr ci -m "Meaningful merge message."

Submit reviewed code
--------------------

::

 cd trunk-lp
 bzr pull lp:nipy
 bzr merge ../trunk-dev
 bzr ci --author:"user name"
 # enter detailed commit message in a docstring format
 bzr push lp:nipy


Shared Repository
-----------------

bzr shared repository?


Example: Merging Fernando's Matplotlib Sphinx docs
--------------------------------------------------

One of the first test cases of the trunk-lp/trunk-dev branch strategy
was merging the matplotlib_ documentation skeleton, which uses sphinx_
into the nipy-trunk.  The summary of the process:

#. Pull the nipy trunk and confirm your main trunk is current.
#. Download Fernando's branch.
#. Build and test Fernando's branch.
#. Merge Fernando's trunk into the main trunk.
#. Build and test main trunk after the merge.
#. Commit merge and push to launchpad

Matthew Brett performed this merge on his machine.

::

    # Change to nipy source repository
    cd ~/dev_trees/nipy-repo

    # Update main trunk
    cd lp-trunk/
    bzr pull

    # Download Fernando branch
    bzr branch lp:~fperez/nipy/trunk-dev fp-trunk-dev
    cd ..

    # Build and test.  # FIXME: Need a good system for doing this
    #python setup.py build
    #python setup.py install
    #python -c "import neuroimaging as ni; ni.test()"

    # Merge Fernando branch
    bzr log -r last:
    bzr merge ../fp-trunk
    bzr commit

    # Push up to launchpad
    bzr push bzr+ssh://matthew-brett@bazaar.launchpad.net/~nipy-developers/nipy/trunk --remember


.. _nipy: https://launchpad.net/nipy
.. _matplotlib: http://matplotlib.sourceforge.net/
.. _sphinx: http://sphinx.pocoo.org/
.. _launchpad: https://launchpad.net/


