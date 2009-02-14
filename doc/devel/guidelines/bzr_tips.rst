.. _bazaar_tips:

=============
 Bazaar Tips
=============

There are a few adjustments in switching from a svn workflow to a bzr
workflow.  This page is meant to catalog some of those gotcha's and
help all of us transition.

Nipy's current workflow
=======================

Most of us are using a distributed workflow process in what bazaar_
calls the *Decentralized with human gatekeeper*.  The `bazaar
documenation on workflows <http://bazaar-vcs.org/Workflows>`_ explains
this process and the various workflows possible with bazaar.

The `Bazaar User Guide
<http://doc.bazaar-vcs.org/latest/en/user-guide/index.html#using-gatekeepers>`_
also has more details.

Pushing code to launchpad
=========================

When you want to push your changes you must specify the shared
mainline to push to.  The ``remember`` flag will make bzr store this
location so later you only need ``$ bzr push``::

  $ bzr push bzr+ssh://USERNAME@bazaar.launchpad.net/~USERNAME/nipy/trunk-dev --remember


.. _bzr_ssh_push_error:

Push Error: Permission Denied (publickey)
=========================================

If you get this error when you try to push your changes you need to
upload your ssh key to Launchpad.  Bazarr has a `SSH page
<http://bazaar-vcs.org/Bzr_and_SSH>`_ explaining some of the details.

Basically you create your public key with ``ssh-keygen``, then login
to your launchpad_ site, click on the *Change details* link and follow
the *Update SSH keys* link in the menu and upload your key.

What changes am I about to submit?
==================================

You've been making changes over several days and want to see what
you've done before merging with the mainline::

  cburns@nipy 12:44:46 $ bzr diff -r submit:
  Using parent branch http://bazaar.launchpad.net/%7Enipy-developers/nipy/trunk/

What was my last commit?
========================

This will print out the log for the last commit.  A meaningful commit
message helps here::

  cburns@nipy 12:44:53 $ bzr log -r last:

Diff between revisions
======================

Diff between two revision numbers::

  cburns@formats 19:53:18 $ bzr diff -r1516..1538 analyze.py

Pipe it into colordiff for colored output::

  cburns@formats 19:53:18 $ bzr diff -r1516..1538 analyze.py | colordiff

.. include:: ../../links_names.txt
