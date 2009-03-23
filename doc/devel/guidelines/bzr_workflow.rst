
.. _bzr_workflow:

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
occasional developer can contribute code to nipy_ since it does not
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

.. _init_trunk_dev:

Initializing your development environment
-----------------------------------------

We're using the `shared repository
<http://bazaar-vcs.org/SharedRepositoryTutorial>`_ feature of Bazaar
and a `Decentralized with human gatekeeper workflows
<http://bazaar-vcs.org/Workflows>`_.  The basic process is as follows:

#. Create an account on launchpad_, if you don't have one already.
   Follow the *Log in / Register* link on the top of the page.

#. Create a shared repository::

     bzr init-repo --trees nipy-repo
     cd nipy-repo

#. Tell bazaar who you are::
     bzr whoami "Barack Obama <bobama@whitehouse.gov>"

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
machine, in which you can develop nipy_ code.  If you get an error
while trying to push, see the :ref:`bzr_ssh_push_error` document for
instructions on launchpad and your ssh key.

.. _daily-development-cycle:

Daily development cycle
-----------------------

You will usually want to keep track of changes in the nipy_ trunk.  To
do this, you will often want to merge the nipy_ trunk into your
development branch::

 cd trunk-dev
 bzr merge lp:nipy
 bzr commit -m 'Merge from trunk'

When editing you will want to push your changes up to your launchpad_
branch::

 cd trunk-dev
 # edit code
 bzr ci -m "Add a meaningful summary of your edits in one complete sentence."
 # repeat
 ...
 # periodically push to lp:~user (like end of day)
 bzr push

Keeping current with the nipy trunk
-----------------------------------

Since we are all doing development in our own launchpad_ branches,
each developer should keep their branch up-to-date with the main trunk
so their branch doesn't diverge significantly from the rest of the
team.  This is done with the ``bzr merge lp:nipy`` line in the
:ref:`daily-development-cycle` section above.  Launchpad allows
developers to subscribe to a branch and receive email notifications
when there are changes.  By subscribing to the `nipy trunk`_, you will
be emailed whenever the trunk changes, reminding you to merge the
trunk into your branch.

On the `nipy trunk`_ page there is an **Owner / Subscribers** section
on the right-hand side.  Click on the link *Subscribe yourself*.
Select the desired options and click the *Subscribe* button.

These options should be sufficient for developers:

  Notification Level:
    Branch attribute and revision notifications

  Generated Diff Size Limit:
    Don't send diffs (*more below*)

  Code review Level:
    Email about all changes

The *Diff Size Limit* is a personal preference.  The full diff's can
be long and a lot to read in an email.  I find it easier to view the
changesets on Launchpad with the color-coded diffs.

**Note:** Developers can also subscribe to other team members branches.

Code review of developer branches
---------------------------------

In order to do code reviews, each developer should propose their
developer branch for merging into the nipy mainline.  We have decided
to keep these proposed merges open indefinitely so the code reviewer
can use Launchpad to view branch changes prior to pulling the branch.
*You only need to do this procedure once.*

In your developer code directory, select the **Propose for merging
into another branch** link and choose these options:

  Target Branch:
    The NIPY mainline (*default selection*)

Click on the **Register** button.

Edit the branch **Title** and **Summary**.  In your branch page,
select the little yellow circle with the pencil icon next to the title
to *Change branch details*.  Set the options to something like:

  Title:
    Chris' development copy of the nipy trunk

  Summary: 
    This branch is my development copy of the trunk. It is
    available here for review by other developers, and merges will be
    periodically made into trunk.

    For this reason, it will always be marked for review to be merged.

  Status:
    Development

Click on the **Change branch** button to finalize.

Look at `Chris' branch
<https://code.launchpad.net/~cburns/nipy/trunk-dev>`_ for an example.

Bazaar Documentation
====================

The `bzr documentation <http://bazaar-vcs.org/Documentation>`_
is thorough and excellent.

Benjamin Thyreau also kindly posted a `bzr tutorial in French
<http://cirl.berkeley.edu/mb312/nipy-docs/tutoriel_bzr.pdf>`_.

And we've start a collection of :ref:`bazaar_tips` for some
common questions.

.. include:: ../../links_names.txt
