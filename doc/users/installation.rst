.. _installation:

====================
Download and Install
====================

This page covers the necessary steps to install and run NIPY.  Below
is a list of required dependencies, along with additional software
recommendations.

NIPY is currently *ALPHA* quality, but is rapidly improving. If you are
trying to get some work done wait until we have a stable release. For now,
the code will primarily be of interest to developers.

Dependencies
------------

Must Have
^^^^^^^^^

  Python_ 2.4 or later
  
  NumPy_ 1.2 or later

  SciPy_ 0.7 or later
    Numpy and Scipy are high-level, optimized scientific computing libraries.

  PyNifti_
    We are using pynifti for the underlying file IO for nifti files.

  gcc_
    NIPY does contain a few C extensions for optimized
    routines. Therefore, you must have a compiler to build from
    source.  XCode_ (OSX) and MinGW_ (Windows) both include gcc.  (*Once
    we have binary packages, this requirement will not be necessary.*)

Strong Recommendations
^^^^^^^^^^^^^^^^^^^^^^

  iPython_
    Interactive python environment.

  Matplotlib_
    2D python plotting library.

Installing from binary packages
-------------------------------

Currently we do not have binary packages.  Until we do, the easiest
installation method is to download the source tarball and follow the
:ref:`building_source` instructions below.

.. _building_source:

Building from source code
-------------------------

Developers should look through the 
:ref:`development quickstart <development-quickstart>` 
documentation.  There you will find information on building NIPY, the
required software packages and our developer guidelines.

If you are primarily interested in using NIPY, download the source
tarball and follow these instructions for building.  The installation
process is similar to other Python packages so it will be familiar if
you have Python experience.

Unpack the tarball and change into the source directory.  Once in the
source directory, you can build the neuroimaging package using::

    python setup.py build

To install, simply do::
   
    sudo python setup.py install

Note: As with any Python_ installation, this will install the modules
in your system Python_ *site-packages* directory (which is why you
need *sudo*).  Many of us prefer to install development packages in a
local directory so as to leave the system python alone.  This is
mearly a preference, nothing will go wrong if you install using the
*sudo* method.  To install in a local directory, use the **--prefix**
option.  For example, if you created a ``local`` directory in your
home directory, you would install nipy like this::

    python setup.py install --prefix=$HOME/local

.. include:: ../links_names.txt

