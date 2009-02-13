================================
 Development install on windows
================================

The easiest way to get the dependencies is to install the `Enthought Tool
Suite`_ .  This gives you MinGW_, Python_, Numpy_, Scipy_, ipython_
and matplotlib_.  If instead you want to do it by component, try the
instructions below.  

Requirements:

* Download and install MinGW_
* Download and install the windows binary for Python_
* Download and install the Numpy_ and Scipy_ binaries

Options:

* Download and install ipython_, being careful to follow the windows
  installation instructions
* Download and install matplotlib_

Alternatively, if you are very brave, you may want to install numpy / scipy from
source - see our maybe out of date :ref:`windows_scipy_build` for details.

Whether you used ETS_ or the instructions above, you will next need to get
the NIPY code via version control:

* Download and install the windows binary for bazaar_
* Go to the windows menu, find the `bazaar` menu, and run ``bzr`` in a
  windows terminal.

You should now be able to follow the instructions in
:ref:`trunk_download`, but with the following modifications:

First, for the ``python setup.py`` steps, you will need to add the
``--compiler=mingw32`` flag, like this::

   python setup.py build_ext --inplace --compiler=mingw32

To build a windows installer, use::

   python setup.py bdist_wininst

Second, you may want to use the ipython_ shell to work in.  For system
 commands use the ``!`` escape, like this, from the ipython prompt::

   !python setup.py build_ext --inplace --compiler=mingw32


.. include:: ../../links_names.txt
