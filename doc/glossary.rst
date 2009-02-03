==========
 Glossary
==========

.. glossary::

   AFNI
      AFNI_ is a functional imaging analysis package.  It is funded by
      the NIMH, based in Bethesda, Maryland, and directed by Robert
      Cox.  Like :term:`FSL`, it is written in C, and it's very common
      to use shell scripting of AFNI command line utilities to
      automate analyses.  Users often describe liking AFNI's
      scriptability, and image visualization. It uses the :term:`GPL`
      license.

   DTI
      Diffusion tensor imaging.  DTI is rather poorly named, because
      it is a model of the diffusion signal, and an analysis method,
      rather than an imaging method.  The simplest and most common
      diffusion tensor model assumes that diffusion direction and
      velocity at every voxel can be modeled by a single tensor - that
      is, by an ellipse of regular shape, fully described by the length
      and orientation of its three orthogonal axes.  This model can
      easily fail in fairly common situations, such as white-matter
      fiber track crossings.

   DWI
      Diffusion-weighted imaging.  DWI is the general term for MRI
      imaging designed to image diffusion processes.  Sometimes
      reseachers use :term:`DTI` to have the same meaning, but
      :term:`DTI` is a common DWI signal model and analysis method.

   EEGlab
      The most widely-used open-source package for analyzing
      electrophysiological data.  EEGlab_ is written in :term:`matlab`
      and uses a :term:`GPL` license.

   FSL
      FSL_ is the FMRIB_ software library, written by the FMRIB_
      analysis group, and directed by Steve Smith.  Like :term:`AFNI`,
      it is a large collection of C / C++ command line utilities that
      can be scripted with a custom GUI / batch system, or using shell
      scripting.  Its particular strength is analysis of :term:`DWI`
      data, and :term:`ICA` functional data analysis, although it has
      strong tools for the standard :term:`SPM approach` to FMRI. It
      is free for academic use, and open-source, but not free for
      commercial use.

   GPL
      The GPL_ is the GNU general public license.  It is one of the most
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
      SPM (statistical parametric mapping) refers either to the
      :term:`SPM approach` to analysis or the :term:`SPM software`
      package.

   SPM approach
       Statistical parametric mapping is a way of analyzing data, that
       involves creating an image (the *map*) containing statistics,
       and then doing tests on this statistic image.  For example, we
       often create a t statistic image where each :term:`voxel`
       contains a t statistic value for the time-series from that
       voxel.  The :term:`SPM software` package implements this
       approach - as do several others, including :term:`FSL` and
       :term:`AFNI`.
      
   SPM software
       SPM_ (statistical parametric mapping) is the name of the
       matlab_ based package written by John Ashburner, Karl Friston
       and others at the `Functional Imaging Laboratory`_ in
       London. More people use the SPM package to analyze :term:`FMRI`
       and :term:`PET` data than any other.  It has good lab and
       community support, and the :term:`matlab` source code is
       available under the :term:`GPL` license.

   voxel
      Voxels are volumetric pixels - that is, they are values in a
      regular grid in three dimensional space - see
      http://en.wikipedia.org/wiki/Voxel


.. include:: links_names.txt
