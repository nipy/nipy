==============================
 Geography of the Scipy world
==============================

in which we briefly describe the various components you are likely to
come across when writing scientific python software in general, and NIPY
code in particular.

Numpy
=====

NumPy_ is the basic Python array-manipulation package.  It allows you to
create, slice and manipulate N-D arrays at near C speed.  It also has
basic arithmetical and mathematical functions (such as sum, mean, and
log, exp, sin, cos), matrix multiplication (``numpy.dot``), Fourier
transforms (``numpy.fft``) and basic linear algebra ``numpy.linalg``.

SciPy
=====

Scipy_ is a large umbrella project that builds on Numpy (and depends on
it).  It includes a variety of high level science and engineering
modules together as a single package.  There are extended modules for
linear algebra (including wrappers to BLAS and LAPACK), optimization,
integration, sparse matrices, special functions, FFTs, signal and image
processing, genetic algorithms, ODE solvers, and others.

Matplotlib
==========

Matplotlib_ is a 2D plotting package that depends on NumPy_.  It has a
simple matlab-like plotting syntax that makes it relatively easy to
create good-looking plots, histograms and images with a small amount of
code.  As well as this simplified Matlab-like syntax, There is also a
more powerful and flexible object-oriented interface.

Ipython
=======

Ipython_ is an interactive shell for python that has various features of
the interactive shell of Matlab, Mathematica and R.  It works
particularly well with Matplotlib_, but is also an essential tool for
interactive code development and code exploration.  It contains
libraries for creainteracting with parallel jobs on clusters or over
several CPU cores in a fairly transparent way.

Cython
======

Cython_ is a development language that allows you to write a combination
of Python and C-like syntax to generate Python extensions.  It is
especially good for linking C libraries to Python in a readable way.  It
is also an excellent choice for optimization of Python code, because it
allows you to drop down to C or C-like code at your bottlenecks without
losing much of the readability of Python.

Mayavi
======

Mayavi_ is a high-level python interface to the VTK_ plotting
libraries.

.. include:: ../links_names.txt
