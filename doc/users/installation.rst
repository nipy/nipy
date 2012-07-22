.. _installation:

====================
Download and Install
====================

This page covers the necessary steps to install and run NIPY.  Below is a list
of required dependencies, along with additional software recommendations.

NIPY is currently *ALPHA* quality, but is rapidly improving.

Dependencies
------------

Must Have
^^^^^^^^^

  Python_ 2.5 or later

  NumPy_ 1.2 or later

  SciPy_ 0.7 or later
    Numpy and Scipy are high-level, optimized scientific computing libraries.

  Sympy_ 0.6.6 or later
    Sympy is a symbolic mathematics library for Python.  We use it for
    statistical formalae.


Must Have to Build
^^^^^^^^^^^^^^^^^^

If your OS/distribution does not provide you with binary build of
NIPY, you would need few additional components to be able to build
NIPY directly from sources

  gcc_
    NIPY does contain a few C extensions for optimized
    routines. Therefore, you must have a compiler to build from
    source.  XCode_ (OSX) and MinGW_ (Windows) both include gcc.

  cython_ 0.11.1 or later
    Cython is a language that is a fusion of Python and C.  It allows us
    to write fast code using Python and C syntax, so that it easier to
    read and maintain.


Strong Recommendations
^^^^^^^^^^^^^^^^^^^^^^

  iPython_
    Interactive Python environment.

  Matplotlib_
    2D python plotting library.


Installing from binary packages
-------------------------------

Currently we have binary packages for snapshot releases only for
Debian-based systems.  Stock Debian_ and Ubuntu_ installations come
with some snapshot of NiPy available.  For more up-to-date packages of
NiPy you can use NeuroDebian_ repository.  For the other OSes and
Linux distributions, the easiest installation method is to download
the source tarball and follow the :ref:`building_source` instructions
below.

.. _building_source:

Building from source code
-------------------------

Developers should look through the
:ref:`development quickstart <development-quickstart>`
documentation.  There you will find information on building NIPY, the
required software packages and our developer guidelines.

If you are primarily interested in using NIPY, download the source
tarball (e.g. from `nipy github`_) and follow these instructions for building.  The installation
process is similar to other Python packages so it will be familiar if
you have Python experience.

Unpack the source tarball and change into the source directory.  Once in the
source directory, you can build the neuroimaging package using::

    python setup.py build

To install, simply do::

    sudo python setup.py install

.. note::

    As with any Python_ installation, this will install the modules
    in your system Python_ *site-packages* directory (which is why you
    need *sudo*).  Many of us prefer to install development packages in a
    local directory so as to leave the system python alone.  This is
    merely a preference, nothing will go wrong if you install using the
    *sudo* method.  To install in a local directory, use the **--prefix**
    option.  For example, if you created a ``local`` directory in your
    home directory, you would install nipy like this::

        python setup.py install --prefix=$HOME/local

Installing useful data files
-----------------------------

See :ref:`data-files` for some instructions on installing data packages.

.. include:: ../links_names.txt
