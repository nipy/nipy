===================================
 Debian / Ubuntu developer install
===================================

Dependencies
------------

See :ref:`installation` for the installation instructions.  Since NiPy
is provided within stock distribution (``main`` component of Debian,
and ``universe`` of Ubuntu), to install all necessary requirements it
is enough to::

    sudo apt-get build-dep python-nipy

.. note::

   Above invocation assumes that you have references to ``Source``
   repository listed with ``deb-src`` prefixes in your apt .list files.

Otherwise, you can revert to manual installation with::

   sudo apt-get build-essential
   sudo apt-get install python-dev
   sudo apt-get install python-numpy python-numpy-dev python-scipy
   sudo apt-get install liblapack-dev
   sudo apt-get install python-sympy


Useful additions
----------------

Some functionality in NiPy requires additional modules::

   sudo apt-get install ipython
   sudo apt-get install python-matplotlib
   sudo apt-get install mayavi2

For getting the code via version control::

   sudo apt-get install git-core

Then follow the instructions at :ref:`trunk_download`.

And for easier control of multiple Python modules installations
(e.g. different versions of IPython)::

   sudo apt-get install virtualenvwrapper
