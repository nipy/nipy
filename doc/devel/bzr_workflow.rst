======================
 nipy bazaar workflow
======================

.. Contents::

Overview
--------

This document describes the standard workflow for nipy development
using bazaar and the launchpad_ hosting service.  It would help very
much in keeping track of shared code if all nipy_ developers adopt
this workflow.  The workflow should be flexible enough that even an
occassional developer can contribute code to nipy since it does not
require all contributors to have commit access to the main trunk.
This particular workflow was developed to achieve these primary
objectives.

#. Maintain a linear revision number history on the main nipy-trunk.
#. Provide a clear process for code reviews.
#. Minimize damage to the trunk during large code transitions.


Launchpad structure
-------------------

On launchpad_, there exists one main nipy_ trunk that only an
administrator has commit access to (see :ref:`bzr_administration`).
All developers have their own launchpad.net/~developer directory which
may be registered with the nipy_ project.  In addition, there may be
shared team directories on nipy_, for feature development, but in
general individual development should happen in your ~developer
directory.

A snapshot of the nipy_ code directories on launchpad_

* lp:nipy - the nipy-trunk
* ~cburns/nipy - developer branch for cburns (registered with nipy)
* ~twaite/nipy - developer branch for twaite (registered with nipy)

Developer directory structure
-----------------------------
Your development directory will contain at least one branch:

* nipy-repo/trunk-dev

where *trunk-dev* is a branch from your user directory on launchpad_
and is your working development branch.

Initializing your development environment
----------------------------------------- 

We're using the `shared repository <http://bazaar-vcs.org/SharedRepositoryTutorial>`_ feature of bazaar_.

#. Create a shared repository::

     bzr init-repo --trees nipy-repo
     cd nipy-repo

#. Create your own personal development branch, named `trunk-dev`, by
   replicating the nipy_ trunk::

     bzr branch lp:nipy trunk-dev

#. Push your development branch up to launchpad_ for backup and to
   make your changes visible to the rest of the team::

     cd trunk-dev
     bzr push bzr+ssh://USER@bazaar.launchpad.net/~USER/nipy/trunk-dev --remember

   Real example of the line above for user `cburns`::

     bzr push bzr+ssh://cburns@bazaar.launchpad.net/~cburns/nipy/trunk-dev --remember

You now have your own branch, stored on launchpad, and on your
machine, in which you can develop nipy_ code.

Daily development cycle
-----------------------

You will usually want to keep track of changes in the nipy_ trunk.  To
do this, you will often want to merge the nipy_ trunk into your
development branch::

 cd trunk-dev
 bzr merge lp:nipy
 bzr commit -m 'Merge from trunk'

When editing you will want to push your changes up to your launchpad_
branch::                ::

 cd trunk-dev
 # edit code
 bzr ci -m "meaningful message"
 # repeat
 ...
 # periodically push to lp:~user (like end of day)
 bzr push


.. _nipy: https://launchpad.net/nipy
.. _launchpad: https://launchpad.net/
.. _bazaar: http://bazaar-vcs.org/

