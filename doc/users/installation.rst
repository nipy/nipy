.. _installation:

####################
Download and Install
####################

*******
Summary
*******

* if you don't have it, install Python using the instructions below;
* if you don't have it, install Pip_ using the instructions below;
* if you don't have them, install NumPy_ >= 1.14 and Scipy
  >= 1.0 using the instructions below;
* install Nipy with something like:

    .. code-block:: bash

        pip3 install --user nipy

.. note::

    These instructions are for Python 3.  If you are using Python 2.7, use
    ``python2`` instead of ``python3`` and ``pip2`` instead of ``pip3``, for
    the commands below.

*******
Details
*******

Install Python, Pip, Numpy and Scipy
====================================

First install Python 3, then install the Python package installer Pip.

Install Python 3 on Linux
-------------------------

We recommend:

* ``sudo apt-get install -y python3 python3-tk`` (Debian, Ubuntu);
* ``sudo dnf install -y python3 python3-tkinter`` (Fedora).

These are the bare minimum installs.  You will almost certainly want to
install the development tools for Python to allow you to compile other
Python packages:

* ``sudo apt-get install -y python3-dev`` (Debian, Ubuntu);
* ``sudo dnf install -y python3-devel`` (Fedora).

Now :ref:`install-pip`.

Install Python 3 on macOS
-------------------------

We recommend you install Python 3.5 or later using Homebrew
(http://brew.sh/):

.. code-block:: bash

    brew install python3

Homebrew is an excellent all-round package manager for macOS that you can use
to install many other free / open-source packages.

Now :ref:`install-pip`.

.. _install-pip:

Install Pip on Linux or macOS
-----------------------------

Pip can install packages into your main system directories (a *system*
install), or into your own user directories (a *user* install).  We strongly
recommend *user* installs.

To get ready for user installs, put the user local install ``bin``
directory on your user's executable program ``PATH``.  First find the location
of the user ``bin`` directory with:

.. code-block:: bash

    python3 -c 'import site; print(site.USER_BASE + "/bin")'

This will give you a result like ``/home/your_username/.local/bin`` (Linux) or
``/Users/your_username/Library/Python/3.5/bin`` (macOS).

Use your favorite text editor to open the ``~/.bashrc`` file (Linux) or
``.bash_profile`` (macOSX) in your home directory.

Add these lines to end of the file:

.. code-block:: bash

    # Put the path to the local bin directory into a variable
    py3_local_bin=$(python3 -c 'import site; print(site.USER_BASE + "/bin")')
    # Put the directory at the front of the system PATH
    export PATH="$py3_local_bin:$PATH"

Save the file, and restart your terminal to load the configuration from your
``~/.bashrc`` (Linux) or ``~/.bash_profile`` (macOS) file.  Confirm that you
have the user install directory in your PATH, with:

.. code-block:: bash

    echo $PATH

Now install the Python package installer Pip into your user directories (see:
`install pip with get-pip.py`_):

.. code-block:: bash

    # Download the get-pip.py installer
    curl -LO https://bootstrap.pypa.io/get-pip.py
    # Execute the installer for Python 3 and a user install
    python3 get-pip.py --user

Check you have the right version of the ``pip3`` command with:

.. code-block:: bash

    which pip3

This should give you something like ``/home/your_username/.local/bin/pip3``
(Linux) or ``/Users/your_username/Library/Python/3.5/bin`` (macOS).

Now :ref:`install-numpy-scipy`.

.. _install-numpy-scipy:

Install Python 3, Pip, NumPy and Scipy on Windows
-------------------------------------------------

It's worth saying here that very few scientific Python developers use Windows,
so if you're thinking of making the switch to Linux or macOS, now you have
another reason to do that.

Option 1: Anaconda
^^^^^^^^^^^^^^^^^^

If you are installing on Windows, you might want to use the Python 3 version of
`Anaconda`_.  This is a large installer that will install many scientific
Python packages, including NumPy and Scipy, as well as Python itself, and Pip,
the package manager.

The machinery for the Anaconda bundle is not completely open-source, and is
owned by a company, Continuum Analytics. If you would prefer to avoid using
the Anaconda installer, you can also use the Python standard Pip installer.

Option 2: Standard install
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you don't have Python / Pip, we recommend the instructions `here
<http://docs.python-guide.org/en/latest/starting/install3/win>`_ to install
them.  You can also install Python / Pip via the Python 3 installer from the
https://python.org website.

If you already have an old Python installation, you don't have Pip, and you
don't want to upgrade, you will need to download and install Pip following the
instructions at `install pip with get-pip.py`_.

Now open a Cmd or Powershell terminal and run:

.. code-block:: bash

    pip3 install --user numpy scipy

Install Nipy
============

Now you have Python and Pip:

.. code-block:: bash

    pip3 install --user nipy

On Windows, macOS, and nearly all Linux versions on Intel, this will install a
binary (Wheel_) package of NiPy.

***************************
Other packages we recommend
***************************

* IPython_: Interactive Python environment;
* Matplotlib_: Python plotting library.

********************************
Building from latest source code
********************************

Dependencies for build
======================

* A C compiler: Nipy does contain a few C extensions for optimized routines.
  Therefore, you must have a compiler to build from source.  Use XCode_ for
  your C compiler on macOS.  On Windows, you will need the Microsoft Visual
  C++ version corresponding to your Python version - see `using MSVC with
  Python <https://matthew-brett.github.io/pydagogue/python_msvc.html>`_.  On
  Linux you should have the packages you need after you install the
  ``python3-dev`` (Debian / Ubuntu) or ``python3-devel`` (Fedora) packages
  using the instructions above;
* Cython_ 0.12.1 or later:  Cython is a language that is a fusion of Python
  and C.  It allows us to write fast code using Python and C syntax, so that
  it is easier to read and maintain than C code with the same functionality;
* Git_ version control software: follow the instructions on the `main git
  website <git_>`_ to install Git on Linux, macOS or Windows.

Procedure
=========

Please look through the :ref:`development quickstart <development-quickstart>`
documentation.  There you will find information on building NIPY, the required
software packages and our developer guidelines.  Then:

.. code-block:: bash

    # install Cython
    pip3 install --user cython

.. code-block:: bash

    # Clone the project repository
    git clone https://github.com/nipy/nipy

to get the latest development version, and:

.. code-block:: bash

    # Build the latest version in-place
    cd nipy
    pip3 install --user --editable .

to install the code in the development tree into your Python path.

****************************
Installing useful data files
****************************

See :ref:`data-files` for some instructions on installing data packages.

.. include:: ../links_names.txt
