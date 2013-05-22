.. _data-files:

######################
Optional data packages
######################

The source code has some very small data files to run the tests with,
but it doesn't include larger example data files, or the all-important
brain templates we all use.  You can find packages for the optional data
and template files at http://nipy.org/data-packages.

If you don't have these packages, then, when you run nipy installation,
you will probably see messages pointing you to the packages you need.

*********************************************
Data package installation as an administrator
*********************************************

The installation procedure, for now, is very basic.  For example, let us
say that you need the 'nipy-templates' package at
http://nipy.org/data-packages/nipy-templates-0.2.tar.gz
. You simply download this archive, unpack it, and then run the standard
``python setup.py install`` on it.  On a unix system this might look
like::

   curl -O http://nipy.org/data-packages/nipy-templates-0.2.tar.gz
   tar zxvf nipy-templates-0.2.tar.gz
   cd nipy-templates-0.2
   sudo python setup.py install

On windows, download the file, extract the archive to a folder using the
GUI, and then, using the windows shell or similar::

   cd c:\path\to\extracted\files
   python setup.py install

*******************************************
Non-administrator data package installation
*******************************************

The simple ugly manual way
==========================

These are instructions for using the command line in Unix.  You can do similar
things from Windows powershell.

* Locate your nipy user directory from the output of this::

    python -c 'import nibabel.data; print(nibabel.data.get_nipy_user_dir())'

  Call that directory ``<nipy-user>``.  Let's imagine that, for you, this is
  ``~/.nipy``.
* If that directory does not exist already, create it, e.g.::

    mkdir ~/.nipy

* Make a directory in ``<nipy-user>`` called ``nipy``, e.g.::

    mkdir ~/.nipy/nipy

* Go to http://nipy.org/data-packages
* Download the latest *nipy-templates* and *nipy-data* packages
* Unpack both these into some directory, e.g.::

    mkdir data
    cd data
    tar zxvf ~/Downloads/nipy-data-0.2.tar.gz
    tar zxvf ~/Downloads/nipy-templates-0.2.tar.gz

* After you have unpacked the templates, you will have a directory called
  something like ``nipy-templates-0.2``.  In that directory you should see a
  subdirectory called ``templates``.  Copy / move / link the ``templates``
  subdirectory into ``<nipy-user>/nipy``, so you now have a directory
  ``<nipy-user>/nipy/templates``.  From unpacking the data, you should also have
  a directory like ``nipy-data-0.2`` with a subdirectory ``data``.  Copy / move
  / link that ``data`` directory into ``<nipy-user>/nipy`` as well.  For
  example::

    cd data
    cp -r nipy-data-0.2/data ~/.nipy/nipy
    cp -r nipy-templates-0.2/templates ~/.nipy/nipy

* Check whether that worked.  Run the following command from the shell::

    python -c 'import nipy.utils; print(nipy.utils.example_data, nipy.utils.templates)'

  It should show something like::

    (<nibabel.data.VersionedDatasource object at 0x101f8e410>, <nibabel.data.VersionedDatasource object at 0x10044b110>)

  If it shows ``Bomber`` objects instead, something is wrong.  Go back and check
  that you have the nipy home directory right, and that you have directories
  ``<nipy-user>/nipy/data`` and ``<nipy-user>/nipy/templates>``, and that each
  of these two directories have a file ``config.ini`` in them.

The more general way
====================

The commands for the sytem install above assume you are installing into the
default system directories.  If you want to install into a custom directory,
then (in python, or ipython, or a text editor) look at the help for
``nibabel.data.get_data_path()`` . There are instructions there for pointing
your nipy installation to the installed data.

On unix
-------

For example, say you installed with::

   cd nipy-templates-0.2
   python setup.py install --prefix=/home/my-user/some-dir

Then you may want to do make a file ``~/.nipy/config.ini`` with the
following contents::

   [DATA]
   path=/home/my-user/some-dir/share/nipy

On windows
----------

Say you installed with (windows shell)::

   cd nipy-templates-0.2
   python setup.py install --prefix=c:\some\path

Then first, find out your home directory::

   python -c "import os; print os.path.expanduser('~')"

Let's say that was ``c:\Documents and Settings\My User``.  Then, make a
new file called ``c:\Documents and Settings\My User\_nipy\config.ini``
with contents::

   [DATA]
   path=c:\some\path\share\nipy
