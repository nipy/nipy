.. -*- rest -*-
.. vim:syntax=rst

.. image:: https://coveralls.io/repos/nipy/nipy/badge.png?branch=master
    :target: https://coveralls.io/r/nipy/nipy?branch=master

.. Following contents should be from LONG_DESCRIPTION in nipy/info.py


====
NIPY
====

Neuroimaging tools for Python.

The aim of NIPY is to produce a platform-independent Python environment for
the analysis of functional brain imaging data using an open development model.

In NIPY we aim to:

1. Provide an open source, mixed language scientific programming environment
   suitable for rapid development.

2. Create software components in this environment to make it easy to develop
   tools for MRI, EEG, PET and other modalities.

3. Create and maintain a wide base of developers to contribute to this
   platform.

4. To maintain and develop this framework as a single, easily installable
   bundle.

NIPY is the work of many people. We list the main authors in the file
``AUTHOR`` in the NIPY distribution, and other contributions in ``THANKS``.

Website
=======

Current information can always be found at the `NIPY project website
<http://nipy.org/nipy>`_.

Mailing Lists
=============

For questions on how to use nipy or on making code contributions, please see
the ``neuroimaging`` mailing list:

    https://mail.python.org/mailman/listinfo/neuroimaging

Please report bugs at github issues:

    https://github.com/nipy/nipy/issues

You can see the list of current proposed changes at:

    https://github.com/nipy/nipy/pulls

Code
====

You can find our sources and single-click downloads:

* `Main repository`_ on Github;
* Documentation_ for all releases and current development tree;
* Download the `current development version`_ as a tar/zip file;
* Downloads of all `available releases`_.

.. _main repository: http://github.com/nipy/nipy
.. _Documentation: http://nipy.org/nipy
.. _current development version: https://github.com/nipy/nipy/archive/master.zip
.. _available releases: http://pypi.python.org/pypi/nipy

Tests
=====

To run nipy's tests, you will need to install the nose_ Python testing
package.  Then::

    python -c "import nipy; nipy.test()"

You can also run nipy's tests with the ``nipnost`` script in the ``tools``
directory of the nipy distribution::

    ./tools/nipnost nipy

``nipnost`` is a thin wrapper around the standard ``nosetests`` program that
is part of the nose package.  Try ``nipnost --help`` to see a large number of
command-line options.

Dependencies
============

To run NIPY, you will need:

* python_ >= 2.6 (tested with 2.6, 2.7, 3.2 through 3.5)
* numpy_ >= 1.6.0
* scipy_ >= 0.9.0
* sympy_ >= 0.7.0
* nibabel_ >= 1.2

You will probably also like to have:

* ipython_ for interactive work
* matplotlib_ for 2D plotting
* mayavi_ for 3D plotting

.. _python: http://python.org
.. _numpy: http://numpy.scipy.org
.. _scipy: http://www.scipy.org
.. _sympy: http://sympy.org
.. _nibabel: http://nipy.org/nibabel
.. _ipython: http://ipython.org
.. _matplotlib: http://matplotlib.org
.. _mayavi: http://code.enthought.com/projects/mayavi/
.. _nose: http://nose.readthedocs.org/en/latest

License
=======

We use the 3-clause BSD license; the full license is in the file ``LICENSE``
in the nipy distribution.
