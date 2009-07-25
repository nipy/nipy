
Shipping data files for `nipy`
===============================

When developing or using nipy, many data files can be useful. We divide
the data files nipy uses into at least 3 categories

#. *test data* - data files required for routine code testing
#. *template data* - data files required for algorithms to function,
   such as templates or atlases
#. *example data* - data files for running examples, or optional tests

The files used for testing are very small data files. They are shipped
with nipy, and live in the code repository. They live in the module path
``nipy.testing.data``.

.. now a comment .. automodule:: nipy.testing

*template data* and *example data* are example of *data packages*.  What
follows is a discussion of the design and use of data packages.

Use cases for data packages
+++++++++++++++++++++++++++

Using the data package
``````````````````````

The programmer will want to use the data something like this:

.. testcode::

   from nipy.utils import make_repo

   template_repo = make_repo('nipy', 'templates')
   fname = template_repo.get_filename('ICBM152', '2mm', 'T1.nii.gz')
   
where ``fname`` will be the absolute path to the template image
``ICBM152/2mm/T1.nii.gz``. 

The programmer can insist on a particular version of a repository:

.. testcode::

   if template_repo.version < 0.4:
      raise ValueError('Need repository version at least 0.4')

If the repository cannot find the data, then:

>>> make_repo('nipy', 'implausible')
Traceback
 ...
IOError

where ``IOError`` gives a helpful warning about why the data was not
found, and how it should be installed.  

Warnings during installation
````````````````````````````

The example data and template data may be important, and it would be
useful to warn the user if NIPY cannot find either of the two sets of
data when installing the package.  Thus::

   python setup.py install

will import nipy after installation to check whether these raise an error:

>>> from nipy.utils import make_repo
>>> template_repo = make_repo('nipy', 'templates')
>>> data_repo = make_repo('nipy', 'data')

and warn the user accordingly, with some basic instructions for how to
install the data.

Finding the data
````````````````

The routine ``make_repo`` will need to be able to find the data that has
been installed.  For the following call:

>>> template_repo = make_repo('nipy', 'templates')

We propose to:

#. Get a list of paths where data is known to be stored with
   ``nipy.data.get_data_path()``
#. For each of these paths, search for directory ``nipy/templates``.  If
   found, and of the correct format (see below), return a repository,
   otherwise raise an Exception

The paths collected by ``nipy.data.get_data_paths()`` will be
constructed from ':' (Unix) or ';' separated strings.  The source of the
strings (in the order in which they will be used in the search above)
are:

#. The value of the ``NIPY_DATA_PATH`` environment variable, if set
#. Possibly, a section = ``DATA``, parameter = ``path`` entry in a
   ``config.ini`` file in ``nipy_dir`` where ``nipy_dir`` is
   ``$HOME/.nipy`` or equivalent.
#. Section = ``DATA``, parameter = ``path`` entries in configuration
   ``.ini`` files, where the ``.ini`` files are found by
   ``glob.glob(os.path.join(etc_dir, '*.ini')`` and ``etc_dir`` is
   ``/etc/nipy`` on Unix, and some suitable equivalent on Windows.

Data package format
```````````````````

The following tree is an example of the kind of pattern we would expect
in a data directory, where the ``nipy-data`` and ``nipy-templates``
packages have been installed::

  <ROOT> 
  `-- nipy
      |-- data
      |   |-- config.ini
      |   `-- placeholder.txt
      `-- templates
          |-- ICBM152
          |   `-- 2mm
          |       `-- T1.nii.gz
          |-- colin27
          |   `-- 2mm
          |       `-- T1.nii.gz
          `-- config.ini

The ``<ROOT>`` directory is the directory that will appear somewhere in
the list from ``nipy.data.get_data_path()``.  The ``nipy`` subdirectory
signifies data for the ``nipy`` package (as opposed to other
NIPY-related packages such as ``pbrain``).  The ``data`` subdirectory of
``nipy`` contains files from the ``nipy-data`` package.  In the
``nipy/data`` or ``nipy/templates`` directories, there is a
``config.ini`` file, that has at least an entry like this::

  [DEFAULT]
  version = 0.1

giving the version of the repository.  

Installing the data
```````````````````

We will use python distutils to install data packages, and the
``data_files`` mechanism to install the data.  On Unix, with the
following command::

   python setup.py install --prefix=/my/prefix

data will go to::

   /my/prefix/share/nipy

For the example above this will result in these subdirectories::

   /my/prefix/share/nipy/nipy/data
   /my/prefix/share/nipy/nipy/templates

because ``nipy`` is both the project, and the package to which the data
relates.

If you install to a particular location, you will need to add that
location to the output of ``nipy.data.get_data_path()`` using one of the mechanisms above, for example, in your system configuration::

   export NIPY_DATA_PATH=/my/prefix/share/nipy

Packaging for distributions
```````````````````````````

For a particular data package - say ``nipy-templates`` - distributions
will want to:

#. Install the data in set location.  The default from ``python setup.py install`` for the data packages will be ``/usr/share/nipy`` on Unix.
#. Point a system installation of NIPY to these data. 

For the latter, the most obvious route is to copy an ``.ini`` file named
for the data package into the NIPY ``etc_dir``.  In this case, on Unix,
we will want a file called ``/etc/nipy/nipy_templates.ini`` with
contents::

   [DATA]
   path = /usr/share/nipy

Creating the packages / releasing new files
```````````````````````````````````````````

The data in the data packages will not be under source control.

The data packages will be available at a central release location.  For
now this will be: http://cirl.berkeley.edu/mb312/nipy-data but we expect
this to change to sourceforge soon.

A package, such as ``nipy-templates-0.1.tar.gz`` will have the following
contents::


  <ROOT>
    |-- setup.py
    |-- README.txt
    |-- MANIFEST.in
    `-- templates
        |-- ICBM152
        |   `-- 2mm
        |       `-- T1.nii.gz
        |-- colin27
        |   `-- 2mm
        |       `-- T1.nii.gz
        `-- config.ini


There should be only one ``nipy/packagename`` directory delivered by a
particular package.  For example, this package installs
``nipy/templates``, but does not contain ``nipy/data``.  

Making a new package tarball is simply:

#. Downloading and unpacking e.g ``nipy-templates-0.1.tar.gz`` to form
   the directory structure above.
#. Making any changes to the directory
#. Running ``setup.py sdist`` to recreate the package.  

The process of making a release should be:

#. Increment the major or minor version number in the ``config.ini`` file
#. Make a package tarball as above
#. Upload to distribution site

There is an example nipy data package ``nipy-examplepkg`` in the
``examples`` directory of the NIPY repository.

