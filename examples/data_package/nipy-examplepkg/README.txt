NIPY ``examplepkg`` data package
--------------------------------

This is a NIPY data package.

To rebuild this package for distribution:

#. Change the files in the ``examplepkg`` directory
#. Increase the minor or major version number in ``examplepkg/config.ini``

Then::

   python setup.py sdist

And upload the generated file for release. 

