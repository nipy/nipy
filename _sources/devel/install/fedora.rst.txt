==========================
 Fedora developer install
==========================

See :ref:`installation`

This assumes a recent Fedora (>=10) version.  It may work for earlier
versions - see :ref:`installation` for requirements.

This page may also hold for Fedora-based distributions such as
Mandriva and Centos.

Run all the ``yum install`` commands as root.

Requirements::

   yum install gcc-c++
   yum install python-devel
   yum install numpy scipy
   yum install sympy
   yum install atlas-devel

Options::

   yum install ipython
   yum install python-matplotlib

For getting the code via version control::

   yum install git-core

Then follow the instructions at :ref:`trunk_download`
