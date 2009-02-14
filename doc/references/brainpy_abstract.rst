.. _brainpy-hbm-abstract:

============================
 BrainPy HBM abstract, 2005
============================

This is the abstract describing the BrainPy / NIPY project from
the `HBM2005 <http://www.humanbrainmapping.org/toronto2005>`_ conference.

BrainPy: an open source environment for the analysis and visualization of human brain data
==========================================================================================

  Jonathan Taylor (1), Keith Worsley (2), Matthew Brett (3), Yann
  Cointepas (4), John Hunter (5), Jarrod Millman (3), Jean-Baptiste
  Poline (4), Fernando Perez (6)

1. Dept. of Statistics, Stanford University, U.S.A.
2. Dept. of Mathematics and Statistics, !McGill University, Canada
3. Department of Neuroscience, University of California, Berkeley, U.S.A
4. Service Hospitalier Frédéric Joliot, France
5. Complex Systems Laboratory, University of Chicago, U.S.A.
6. Department of Applied Mathematics, University of Colorado at Boulder, U.S.A.

Objective
---------

What follows are the goals of BrainPy, a multi-center project to
provide an open source environment for the analysis and visualization
of human brain data built on top of python. While the project is still
in its initial stages, packages for file I/O, script support as well
as single subject fMRI and random effects group comparisons model are
currently available.

Methods
-------

Scientific computing has evolved over the last two decades in two
broad directions. One, there has been a movement to the use of
high-level interface languages that glue existing high-performance
libraries into an accessible, scripted, interactive environment, eg
IDL, matlab. Two, there has been a shift to open algorithms and
software because this development process leads to better code, and
because it more consistent with the scientific method.

Results & Discussion
--------------------

The proposed environment includes the following:

* We intend to provide users with an open source environment which is
  interoperable with current packages such as SPM and AFNI, both at a
  file I/O level and, where possible, interactively (e.g. pymat --
  calling matlab/SPM from python).
* Read/write/conversion support for all major imaging formats and
  packages (SPM/ANALYZE, :term:`FSL`, :term:`AFNI`, MINC, NIFTI, and
  :term:`VoxBo`
* Low-level access to data through an interactive shell, which is
  important for developing new analysis methods, as well as
  high-level access through GUIs for specialized tasks using standard
  python tools.
* Visualization of results using pre-existing tools such as
  :term:`BrainVisa`, as well as support for development of new tools
  using VTK.
* Support for MATLAB style numeric packages (Numarray) and plotting
  (matplotlib_).
* Support for EEG analysis including EEG/MEG/fMRI fusion analysis.
* Support for spatio-temporal wavelet analysis 
  (`PhiWave <http://phiwave.sourceforge.net>`_)

Conclusions
-----------

BrainPy is an open-source environment for the analysis and
visualization of neuroimaging data built on top of python.

.. include:: ../links_names.txt
