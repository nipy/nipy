==========
 Glossary
==========

.. glossary::

   EEGlab
      The most widely-used open-source package for analyzing
      electrophysiological data.  EEGlab_ is written in :term:`matlab`
      and uses a :term:`GPL` license.

   GPL
      The GNU general public license:
      http://www.gnu.org/licenses/gpl.html.  It is one of the most
      commonly-used open-source sofware licenses.  The distinctive
      feature of the GPL license is that it requires that any code
      derived from GPL code also uses a GPL license.  See also:
      http://en.wikipedia.org/wiki/GNU_General_Public_License

   Matlab
      matlab_ began as a high-level programming language for working
      with matrices.  Over time it has expanded to become a fairly
      general-purpose language.  See also:
      http://en.wikipedia.org/wiki/MATLAB.  It has good numerical
      algorithms, 2D graphics, and documentation.  There are several
      large neuroscience software projects wtitten in matlab,
      including :term:`SPM software`, and :term:`EEGlab`.

   SPM
      SPM (statistical parametric mapping) refers to either the
      :term:`SPM analysis approach` or the :term:`SPM software`

   SPM analysis approach
       Statistical parametric mapping is a way of analyzing data, that
       involves creating an image (the *map*) containing statistics.
       For example, we often create a t statistic image where each
       :term:`voxel` contains a t statistic value for the time-series
       from the voxel.
      
   SPM software
       SPM_ (statistical parametric mapping) software is the name of
       the matlab_ based package written by John Ashburner, Karl
       Friston and others at the `Functional Imaging Laboratory`_ in
       London. More people use the SPM package to analyze :term:`FMRI`
       and :term:`PET` data than any other.  It has good lab and
       community support, and the :term:`matlab` source code is
       available under the :term:`GPL` license.  

   voxel
      Voxels are volumetric pixels - that is, they are values in a
      regular grid in three dimensional space - see
      http://en.wikipedia.org/wiki/Voxel


.. include:: links_names.txt
