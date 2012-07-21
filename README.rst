.. -*- rest -*-
.. vim:syntax=rest

===============================
 Neuroimaging in Python (NIPY)
===============================

A package for processing of functional imaging data. Among the functions
included are:

* Combined motion correction and slice timing
* Flexible affine image registration
* Brain image segmentation
* Smoothing
* Resampling
* Statistical analysis
* Multiple comparison correction

Jonathan Taylor began NIPY as a port and rewrite of `fmristat`_ by `Keith
Worlsey`_.  Later we joined coding efforts with Bertrand Thirion, Alexis Roche
and others at `Neurospin` near Paris, France.  Current NIPY is the work of many
authors and maintainers.  We list the main authors in the file ``AUTHOR`` in the
NIPY distribution.

.. _fmristat: http://www.math.mcgill.ca/keith/fmristat
.. _Keith Worsley: http://www.math.mcgill.ca/keith

Website
=======

Current information can always be found at the NIPY website::

    http://nipy.org/nipy

Mailing Lists
=============

Please see the developer's list::

    http://projects.scipy.org/mailman/listinfo/nipy-devel

Code
====

You can find our sources and single-click downloads:

* `Main repository`_ on Github.
* Documentation_ for all releases and current development tree.
* Download as a tar/zip file the `current trunk`_.
* Downloads of all `available releases`_.

.. _main repository: http://github.com/nipy/nipy
.. _Documentation: http://nipy.org/nipy
.. _current trunk: http://github.com/nipy/nipy/archives/master
.. _available releases: http://github.com/nipy/nipy/downloads

Dependencies
============

To run NIPY, you will need:

* python_ >= 2.5.  We don't yet run on python 3, sad to say.
* numpy_ >= 1.2
* scipy_ >= 0.7.0
* sympy_ >= 0.6.6
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
.. _ipython: http://ipython.scipy.org
.. _matplotlib: http://matplotlib.sourceforge.net
.. _mayavi: http://code.enthought.com/projects/mayavi/

License
=======

We use the 3-clause BSD license; the full license is in the file ``LICENSE`` in
the nipy distribution.
