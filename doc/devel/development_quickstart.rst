.. _development-quickstart:

========================
 Development quickstart
========================

Source Code
===========

NIPY uses github_ for our code hosting.  For immediate access to
the source code, see the `nipy github`_ site.

Checking out the latest version
===============================

To check out the latest version of nipy you need git_::

    git clone git://github.com/nipy/nipy.git

There are two methods to install a development version of nipy.  For
both methods, build the extensions in place::

    python setup.py build_ext --inplace

Then you can either:

#. Create a symbolic link in your *site-packages* directory to the inplace
   build of your source.  The advantage of this method is it does not require
   any modifications of your PYTHONPATH.

#. Place the source directory in your PYTHONPATH.

With either method, all of the modifications made to your source tree
will be picked up when nipy is imported.

.. _installing-data:

Optional data packages
======================

The source code has some very small data files to run the tests with,
but it doesn't include larger example data files, or the all-important
brain templates we all use.  You can find packages for the optional data
and template files at http://nipy.sourceforge.net/data-packages.

If you don't have these packages, then, when you run nipy installation,
you will probably see messages pointing you to the packages you need.

Data package installation as an administrator
---------------------------------------------

The installation procedure, for now, is very basic.  For example, let us
say that you need the 'nipy-templates' package at
http://nipy.sourceforge.net/data-packages/nipy-templates-0.2.tar.gz
. You simply download this archive, unpack it, and then run the standard
``python setup.py install`` on it.  On a unix system this might look
like::

   curl -O http://nipy.sourceforge.net/data-packages/nipy-templates-0.2.tar.gz
   tar zxvf nipy-templates-0.2.tar.gz
   cd nipy-templates-0.2
   sudo python setup.py install

On windows, download the file, extract the archive to a folder using the
GUI, and then, using the windows shell or similar::

   cd c:\path\to\extracted\files
   python setup.py install

Non-administrator data package installation
-------------------------------------------

The commands above assume you are installing into the default system
directories.  If you want to install into a custom directory, then (in python,
or ipython, or a text editor) look at the help for
``nibabel.data.get_data_path()`` . There are instructions there for pointing
your nipy installation to the installed data.

On unix
~~~~~~~

For example, say you installed with::

   cd nipy-templates-0.2
   python setup.py install --prefix=/home/my-user/some-dir

Then you may want to do make a file ``~/.nipy/config.ini`` with the
following contents::

   [DATA]
   /home/my-user/some-dir/share/nipy

On windows
~~~~~~~~~~

Say you installed with (windows shell)::

   cd nipy-templates-0.2
   python setup.py install --prefix=c:\some\path

Then first, find out your home directory::

   python -c "import os; print os.path.expanduser('~')"

Let's say that was ``c:\Documents and Settings\My User``.  Then, make a
new file called ``c:\Documents and Settings\My User\_nipy\config.ini``
with contents::

   [DATA]
   c:\some\path\share\nipy

Guidelines
==========

We have adopted many developer guidelines in an effort to make
development easy, and the source code readable, consistent and robust.
Many of our guidelines are adopted from the scipy_ / numpy_ community.
We welcome new developers to the effort, if you're interested in
developing code or documentation please join the `nipy mailing list`_
and introduce yourself.  If you plan to do any code development, we
ask that you take a look at the following guidelines.  We do our best
to follow these guidelines ourselves:

* :ref:`howto_document` : Documentation is critical.  This document
  describes the documentation style, syntax, and tools we use.

* `Numpy/Scipy Coding Style Guidelines:
  <http://projects.scipy.org/scipy/numpy/wiki/CodingStyleGuidelines>`_
  This is the coding style we strive to maintain.

* :ref:`development-workflow` : This describes our process for version control.

* :ref:`testing` : We've adopted a rigorous testing framework.

* :ref:`optimization`: "premature optimization is the root of all
  evil."

.. _trunk_download:

Submitting a patch
==================

The preferred method to submit a patch is to create a branch of nipy on
your machine, modify the code and make a patch or patches.  Then email
the `nipy mailing list`_ and we will review your code and hopefully
apply (merge) your patch.  See the instructions for
:ref:`making-patches`.

If you do not wish to use git and github, please feel free to
file a bug report and submit a patch or email the 
`nipy mailing list`_.

Bug reports
===========

If you find a bug in nipy, please submit a bug report at the `nipy
bugs`_ github site so that we can fix it.


.. include:: ../links_names.txt
