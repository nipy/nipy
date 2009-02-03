.. _pynifti-io:

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
database.  When you switch branches (``git checkout branch-name``),
git will apply the branch diff to the directory, updating any files to
match the new branch.

Git uses man pages for it's installed documentation.  As with all man
pages, these contain a lot of useful information, so you should know
how to access them.  All git commands can be called in two forms:

1. git add <filename>

2. git-add <filename>

The first form is the one you will probably use most and is what is
often shown in the documentation.  The second form, however, is what
you need to access the man page.  To see the man page on how to add a
file to a git repository::

  man git-add

To see a list of all git commands look at the main git man page::

  man git

As with Bazaar, you should identify yourself to git so the repository
keeps track of who made your edits::

  git config --global user.name "Your Name Comes Here"
  git config --global user.email you@yourdomain.example.com

To list your git configuration::

    cburns@pynifti 13:32:31 $ git config -l
    user.name=Christopher Burns
    user.email=cburns[at]berkeley[dot]edu
    color.diff=auto
    color.status=auto
    core.repositoryformatversion=0
    core.filemode=true
    core.bare=false
    core.logallrefupdates=true
    remote.origin.url=ssh://git.debian.org/git/pkg-exppsy/pynifti.git
    remote.origin.fetch=+refs/heads/*:refs/remotes/origin/*
    branch.master.remote=origin
    branch.master.merge=refs/heads/master

We can also see the remote origin branch from which we have cloned.

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

Example:
^^^^^^^^

I'm in my pynifti source directory::

  cburns@pynifti 13:20:35 $ pwd
  /Users/cburns/src/pynifti

Use the ``git branch`` command without arguments to see all of your
local branches.  Below we can see that I'm in my ``cb/master``
branch::
  
  cburns@pynifti 13:20:39 $ git branch
  * cb/master
    cb/nipy
    master

I want to switch to the ``master`` branch and update it with the
server::

  cburns@pynifti 13:26:08 $ git checkout master
  Switched to branch "master"
  cburns@pynifti 13:26:16 $ git branch
    cb/master
    cb/nipy
  * master

Pull from the server to update our master branch::

  cburns@pynifti 13:35:52 $ git pull
  Password: 
  Already up-to-date.

Switch into ``cb/master`` and merge with the ``master`` branch.
Remember, these are not separate directories, git *knows* about the
other branch by name, so we do not specify a path, we specify a branch
name.::

  cburns@pynifti 13:36:18 $ git checkout cb/master
  Switched to branch "cb/master"

  cburns@pynifti 13:38:38 $ git merge master
  Already up-to-date.

To update the nipy source:
^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Change to the pynifti directory in the nipy developer trunk::

    cd trunk-dev/neuroimaging/externals/pynifti/utils

#. Run the ``update_source.py`` script to update the source.  This
   assumes a directory structure the pynifti sources are in the
   directory ``$HOME/src/pynifti``.  Run the script::

    python update_source.py


.. include:: ../links_names.txt
