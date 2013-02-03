.. _installation:

####################
Download and Install
####################

This page covers the necessary steps to install and run NIPY.  Below is a list
of required dependencies, along with additional software recommendations.

************************
Dependencies for install
************************

Must Have
=========

* Python_ 2.5 or later
* NumPy_ 1.2 or later:  Numpy is an array library for Python
* SciPy_ 0.7 or later:  Scipy contains scientific computing libraries based on
  numpy
* Sympy_ 0.6.6 or later: Sympy is a symbolic mathematics library for Python.  We
  use it for statistical formulae.

Strong Recommendations
======================

* IPython_: Interactive Python environment.
* Matplotlib_: python plotting library.

Installing from binary packages
===============================

For Debian or Ubuntu
--------------------

Please use the NeuroDebian_ repository, and install with::

    sudo apt-get install python-nipy

This will install the dependencies for you.

For Fedora, CentOS
------------------

::

    sudo yum install numpy scipy sympy python-setuptools
    sudo yum install python-devel gcc
    sudo easy_install nibabel
    sudo easy_install nipy

For OSX
^^^^^^^

Install Python, Numpy, and Scipy via their respective ``dmg`` installers.

Install via distribute_ / setuptools_ and ``easy_install``. See the distribute_
page for how to install ``easy_install`` and related tools. Then (from the
command prompt)::

    easy_install nipy

For Windows
^^^^^^^^^^^

Option 1
""""""""

You can make your life much easier by using `Python (X, Y)`_.  This will install
Python, Numpy, Scipy, IPython, Matplotlib, Sympy and many other useful things.

Then go to `nipy pypi`_ and download the ``.exe`` installer for nipy.  Double click
to install.

Option 2
""""""""

* Download Python_ and install with the ``exe`` or ``msi`` installer
* Download and install the "Scipy stack" from Christophe Gohlke's `unofficial
  windows binaries`_.
* If the nipy version on the `unofficial windows binaries`_ page is current, use
  that, otherwise, go to `nipy pypi`_, download and install the ``exe``
  installer for nipy

Option 3
""""""""

Consider one of the big Python bundles such as `EPD free`_ or `Anaconda CE`_ for
the dependencies.  Install nipy from the ``exe`` installer at `nipy pypi`_.

Option 4
""""""""

Do all the installs by hand:

* Download Python_ and install with the ``exe`` or ``msi`` installer.  Make sure
  your python and the scripts directory (say, ``c:\Python27\Scripts``) are on
  your windows path.
* Download Numpy and Scipy ``exe`` installers for your Python version from their
  respective Numpy and Scipy download sites.
* Install distribute_ to give you ``easy_install``.
* Install pip_ using ``easy_install`` from a windows ``cmd`` shell::

    easy_install pip

* Install sympy and nibabel using pip from a window ``cmd`` shell::

    pip install sympy
    pip install nibabel

* On 32-bit Windows, install nipy using ``easy_install``::

    easy_install nipy

  This will pick up and use the ``exe`` installer.  For 64-bits install use the
  installer at the `unofficial windows binaries`_ site.

Otherwise
^^^^^^^^^

I'm afraid you might need to build from source...

.. _building_source:

*************************
Building from source code
*************************

Dependencies for build
======================

* A C compiler: NIPY does contain a few C extensions for optimized routines.
  Therefore, you must have a compiler to build from source.  XCode_ (OSX) and
  MinGW_ (Windows) both include a C compiler.  On Linux, try ``sudo apt-get
  build-essential`` on Debian / Ubuntu, ``sudo yum install gcc`` on Fedora and
  related distributions.

Recommended for build
=====================

* Cython_ 0.12.1 or later:  Cython is a language that is a fusion of Python and
  C.  It allows us to write fast code using Python and C syntax, so that it
  easier to read and maintain. You don't need it to build a release, unless you
  modify the Cython ``*.pyx`` files in the nipy distribution.

Procedure
=========

Developers should look through the
:ref:`development quickstart <development-quickstart>`
documentation.  There you will find information on building NIPY, the
required software packages and our developer guidelines.

If you are primarily interested in using NIPY, download the source
tarball from `nipy pypi` and follow these instructions for building.  The
installation process is similar to other Python packages so it will be familiar
if you have Python experience.

Unpack the source tarball and change into the source directory.  Once in the
source directory, you can build the neuroimaging package using::

    python setup.py build

To install, simply do::

    sudo python setup.py install

.. note::

    As with any Python installation, this will install the modules in your
    system Python *site-packages* directory (which is why you need *sudo*).
    Many of us prefer to install development packages in a local directory so as
    to leave the system python alone.  This is merely a preference, nothing will
    go wrong if you install using the *sudo* method.

    If you have Python 2.6 or later, you might want to do a `user install
    <http://docs.python.org/2/install/index.html#alternate-installation-the-user-scheme>`_

        python setup.py install --user

    To install nipy in some other local directory, use the **--prefix** option.
    For example, if you created a ``local`` directory in your home directory,
    you would install nipy like this::

        python setup.py install --prefix=$HOME/local


Installing useful data files
-----------------------------

See :ref:`data-files` for some instructions on installing data packages.

.. include:: ../links_names.txt
