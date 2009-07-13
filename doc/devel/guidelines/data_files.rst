
Shipping data files for `nipy`
===============================

NeuroImaging is all about data analysis. When developing or using nipy,
many data files can be useful. We divide the data files nipy uses in 3:

 1) Data files required for testing. 

 2) Data files required for algorithm to function, such as templates or
    atlas.

 3) Data files for running examples.

Test files
------------

The files used for testing are very small data files, and they are
shipped with nipy, in the bzr tree. They live in the module path
nipy.testing.data.

.. automodule:: nipy.testing

Templates and altas
--------------------

The core data files required for nipy to function are shipped in a
separate tarball that is expanded in a location specified in the site.cfg
at build time.

These files can be retrieved at run-time using `get_data_file`:

.. autofunction:: nipy.utils.get_data_file

____

You can check what files are available with::

    import os
    from nipy.utils.data_files import data_dir
    os.listdir(data_dir)

Example files
--------------

`nipy` provides an utility function to grab some data from the net,
download it, and cache it locally for running examples:
`get_example_file`:

.. autofunction:: nipy.utils.get_example_file

____

Similarly, the local cache directory is specified in the site.cfg at
install time, and can be inspected using::

    import os
    from nipy.utils.data_files import example_data_dir
    os.listdir(example_data_dir)



